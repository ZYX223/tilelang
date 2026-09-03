"""Rewrite canonical scalar GEMM TIR to an abstract Sunway FP32x8 FMA."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tvm import IRModule, ir, tirx

from .plan import SunwayGemmPlan


SEMANTIC_FMA_F32X8 = "tilelang_sunway_fma_f32x8"
NATIVE_FMA_F32X8 = "tilelang_sunway_native_fma_f32x8"


@dataclass(frozen=True, slots=True)
class _ScalarGemmRegion:
    m_loop: tirx.For
    n_loop: tirx.For
    k_loop: tirx.For
    a_load: tirx.BufferLoad
    b_load: tirx.BufferLoad
    c_load: tirx.BufferLoad


def _map_prim_funcs(
    mod: IRModule,
    rewrite: Callable[[tirx.PrimFunc], tirx.PrimFunc],
) -> IRModule:
    functions = {
        global_var: rewrite(func) if isinstance(func, tirx.PrimFunc) else func
        for global_var, func in mod.functions.items()
    }
    return IRModule(functions, attrs=mod.attrs, global_infos=mod.global_infos)


def _static_loop(loop: object, extent: int) -> bool:
    return (
        isinstance(loop, tirx.For)
        and loop.kind == tirx.ForKind.SERIAL
        and isinstance(loop.min, tirx.IntImm)
        and int(loop.min) == 0
        and isinstance(loop.extent, tirx.IntImm)
        and int(loop.extent) == extent
    )


def _buffer_shape(buffer: tirx.Buffer) -> tuple[int, ...] | None:
    if any(not isinstance(dim, tirx.IntImm) for dim in buffer.shape):
        return None
    return tuple(int(dim) for dim in buffer.shape)


def _same_buffer(left: tirx.Buffer, right: tirx.Buffer) -> bool:
    return left.data.same_as(right.data)


def _match_scalar_gemm_region(node: object, plan: SunwayGemmPlan) -> _ScalarGemmRegion | None:
    if not _static_loop(node, plan.tile_m):
        return None
    m_loop = node
    if not _static_loop(m_loop.body, plan.tile_n):
        return None
    n_loop = m_loop.body
    if not _static_loop(n_loop.body, plan.tile_k):
        return None
    k_loop = n_loop.body
    if not isinstance(k_loop.body, tirx.BufferStore):
        return None
    store = k_loop.body
    if not isinstance(store.value, tirx.Add):
        return None

    add_operands = (store.value.a, store.value.b)
    c_load = next(
        (
            value
            for value in add_operands
            if isinstance(value, tirx.BufferLoad)
            and _same_buffer(value.buffer, store.buffer)
            and ir.structural_equal(value.indices, store.indices)
        ),
        None,
    )
    product = next((value for value in add_operands if isinstance(value, tirx.Mul)), None)
    if c_load is None or product is None:
        return None
    if not isinstance(product.a, tirx.BufferLoad) or not isinstance(product.b, tirx.BufferLoad):
        return None

    loads = (product.a, product.b)
    a_load = next(
        (load for load in loads if _buffer_shape(load.buffer) == (plan.tile_m, plan.tile_k)),
        None,
    )
    b_load = next(
        (load for load in loads if _buffer_shape(load.buffer) == (plan.tile_k, plan.tile_n)),
        None,
    )
    if a_load is None or b_load is None or a_load.same_as(b_load):
        return None

    expected_a = [m_loop.loop_var, k_loop.loop_var]
    expected_b = [k_loop.loop_var, n_loop.loop_var]
    if not ir.structural_equal(a_load.indices, expected_a) or not ir.structural_equal(
        b_load.indices, expected_b
    ):
        return None
    return _ScalarGemmRegion(m_loop, n_loop, k_loop, a_load, b_load, c_load)


def _substitute_indices(
    indices: list[tirx.PrimExpr],
    replacements: dict[tirx.Var, tirx.PrimExpr],
) -> list[tirx.PrimExpr]:
    return [tirx.stmt_functor.substitute(index, replacements) for index in indices]


def _flat_index(buffer: tirx.Buffer, indices: list[tirx.PrimExpr]) -> tirx.PrimExpr:
    if len(indices) != len(buffer.shape) or not indices:
        raise ValueError("Sunway G1 SIMD requires a non-scalar compact buffer access")
    if any(not isinstance(dim, tirx.IntImm) for dim in buffer.shape):
        raise ValueError("Sunway G1 SIMD requires static compact buffer shapes")
    offset = indices[0]
    for index, dim in zip(indices[1:], buffer.shape[1:], strict=True):
        offset = offset * int(dim) + index
    return offset


def _build_simd_region(region: _ScalarGemmRegion, plan: SunwayGemmPlan) -> tirx.For:
    m = tirx.Var("sunway_simd_m", "int32")
    k = tirx.Var("sunway_simd_k", "int32")
    n_vector = tirx.Var("sunway_simd_n_vector", "int32")
    n = n_vector * plan.vector_width
    replacements = {
        region.m_loop.loop_var: m,
        region.n_loop.loop_var: n,
        region.k_loop.loop_var: k,
    }

    a_indices = _substitute_indices(list(region.a_load.indices), replacements)
    b_indices = _substitute_indices(list(region.b_load.indices), replacements)
    c_indices = _substitute_indices(list(region.c_load.indices), replacements)
    a_value = tirx.BufferLoad(region.a_load.buffer, a_indices)
    b_pointer = region.b_load.buffer.access_ptr(
        "r",
        offset=_flat_index(region.b_load.buffer, b_indices),
        extent=plan.vector_width,
    )
    c_pointer = region.c_load.buffer.access_ptr(
        "rw",
        offset=_flat_index(region.c_load.buffer, c_indices),
        extent=plan.vector_width,
    )
    fma = tirx.Evaluate(
        tirx.call_extern(
            "int32",
            SEMANTIC_FMA_F32X8,
            a_value,
            b_pointer,
            c_pointer,
        )
    )
    n_loop = tirx.For(
        n_vector,
        0,
        plan.tile_n // plan.vector_width,
        tirx.ForKind.SERIAL,
        fma,
    )
    k_loop = tirx.For(k, 0, plan.tile_k, tirx.ForKind.SERIAL, n_loop)
    return tirx.For(m, 0, plan.tile_m, tirx.ForKind.SERIAL, k_loop)


def lower_gemm_compute_to_simd(mod: IRModule, plan: SunwayGemmPlan) -> IRModule:
    """Replace exactly one canonical scalar tile computation with FP32x8 semantics."""

    if plan.compute != "simd" or plan.vector_width != 8:
        raise ValueError("Sunway G1 SIMD lowering requires compute='simd' and vector width 8")

    def rewrite_func(func: tirx.PrimFunc) -> tirx.PrimFunc:
        sites = 0

        def rewrite(node: object) -> object | None:
            nonlocal sites
            region = _match_scalar_gemm_region(node, plan)
            if region is None:
                return None
            sites += 1
            return _build_simd_region(region, plan)

        body = tirx.stmt_functor.ir_transform(func.body, None, rewrite)
        if sites != 1:
            raise ValueError(
                "Sunway G1 SIMD requires exactly one canonical C += A * B loop nest; "
                f"found {sites}"
            )
        return func.with_body(body)

    return _map_prim_funcs(mod, rewrite_func)
