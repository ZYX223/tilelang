"""Static schedule extraction for canonical tiled Sunway GEMM programs."""

from __future__ import annotations

from dataclasses import dataclass

from tvm import tirx

from tilelang.sunway.target import SunwayTargetConfig


_SW64_POINTER_BYTES = 8
_REPLY_BYTES = 4


def _call_name(call: tirx.Call) -> str:
    return str(getattr(call.op, "name", ""))


def _static_int(value: tirx.PrimExpr, description: str) -> int:
    if not isinstance(value, tirx.IntImm):
        raise ValueError(f"Sunway G0 requires a static {description}, got {value}")
    result = int(value)
    if result <= 0:
        raise ValueError(f"Sunway G0 requires a positive {description}, got {result}")
    return result


def _region_buffer(value: tirx.PrimExpr, role: str) -> tirx.Buffer:
    if not isinstance(value, tirx.Call) or _call_name(value) != "tl.region":
        raise ValueError(f"Sunway G0 GEMM {role} must be a direct TileLang buffer region")
    base = value.args[0]
    if not isinstance(base, tirx.BufferLoad):
        raise ValueError(f"Sunway G0 GEMM {role} region must start from a buffer load")
    return base.buffer


def _thread_loops(func: tirx.PrimFunc) -> dict[str, tirx.For]:
    result: dict[str, tirx.For] = {}

    def visit(node: object) -> None:
        if not isinstance(node, tirx.For) or node.thread_binding is None:
            return
        tag = str(node.thread_binding.thread_tag)
        if tag in result:
            raise ValueError(f"Sunway G0 requires one {tag} binding per PrimFunc")
        result[tag] = node

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return result


@dataclass(frozen=True, slots=True)
class SunwayGemmPlan:
    """Static tile, mesh, compute, dtype, and LDM schedule decisions."""

    tile_m: int
    tile_n: int
    tile_k: int
    block_tiles_m: int
    block_tiles_n: int
    k_panels: int
    global_m: int
    global_n: int
    global_k: int
    workers: int
    cpe_rows: int
    cpe_cols: int
    active_cpes: int
    ownership: str
    compute: str
    vector_width: int
    stages: int
    input_dtype: str
    accum_dtype: str
    ldm_bytes: int

    @classmethod
    def from_prim_func(
        cls,
        func: tirx.PrimFunc,
        config: SunwayTargetConfig,
    ) -> SunwayGemmPlan:
        gemm_calls: list[tirx.Call] = []

        def collect_gemm(node: object) -> None:
            if isinstance(node, tirx.Call) and _call_name(node) == "tl.tileop.gemm":
                gemm_calls.append(node)

        tirx.stmt_functor.post_order_visit(func.body, collect_gemm)
        if len(gemm_calls) != 1:
            raise ValueError(f"Sunway G0 requires exactly one T.gemm site; found {len(gemm_calls)}")
        gemm = gemm_calls[0]

        tile_m = _static_int(gemm.args[5], "GEMM M tile")
        tile_n = _static_int(gemm.args[6], "GEMM N tile")
        tile_k = _static_int(gemm.args[7], "GEMM K tile")
        a_buffer = _region_buffer(gemm.args[0], "A")
        b_buffer = _region_buffer(gemm.args[1], "B")
        c_buffer = _region_buffer(gemm.args[2], "C")
        input_dtype = str(a_buffer.dtype)
        accum_dtype = str(c_buffer.dtype)
        if input_dtype != "float32" or str(b_buffer.dtype) != "float32" or accum_dtype != "float32":
            raise ValueError("Sunway G0 GEMM requires FP32 A, B, and C buffers")

        for param, buffer in func.buffer_map.items():
            if param not in func.params:
                continue
            if len(buffer.shape) != 2 or any(not isinstance(extent, tirx.IntImm) for extent in buffer.shape):
                raise ValueError("Sunway G0 GEMM requires static two-dimensional argument buffers")

        bindings = _thread_loops(func)
        if "threadIdx.x" not in bindings:
            raise ValueError("Sunway G0 GEMM requires a threadIdx.x worker binding")
        workers = _static_int(bindings["threadIdx.x"].extent, "logical worker count")
        expected_workers = config.cpe_rows * config.cpe_cols
        if workers != expected_workers:
            raise ValueError(f"Sunway G0 requires {expected_workers} logical workers, got {workers}")
        for tag in ("threadIdx.y", "threadIdx.z", "blockIdx.z"):
            if tag in bindings and _static_int(bindings[tag].extent, f"{tag} extent") != 1:
                raise ValueError(f"Sunway G0 requires {tag} extent 1")
        for tag in ("blockIdx.x", "blockIdx.y"):
            if tag not in bindings:
                raise ValueError(f"Sunway G0 GEMM requires a {tag} tile binding")
        block_tiles_n = _static_int(bindings["blockIdx.x"].extent, "blockIdx.x extent")
        block_tiles_m = _static_int(bindings["blockIdx.y"].extent, "blockIdx.y extent")

        pipeline_values: list[tuple[int, int]] = []

        def collect_stages(node: object) -> None:
            if not isinstance(node, tirx.For) or "num_stages" not in node.annotations:
                return
            names: list[str] = []
            tirx.stmt_functor.post_order_visit(
                node.body,
                lambda child: names.append(_call_name(child)) if isinstance(child, tirx.Call) else None,
            )
            if "tl.tileop.gemm" in names:
                pipeline_values.append(
                    (
                        _static_int(node.extent, "GEMM K-panel count"),
                        int(node.annotations["num_stages"]),
                    )
                )

        tirx.stmt_functor.post_order_visit(func.body, collect_stages)
        if len(pipeline_values) != 1:
            raise ValueError("Sunway G0 requires exactly one annotated GEMM pipeline loop")
        k_panels, stages = pipeline_values[0]
        if stages != 1:
            raise ValueError(f"Sunway G0 requires num_stages=1, got {stages}")

        vector_width = config.simd_width if config.gemm_compute == "simd" else 1
        if config.gemm_compute == "simd" and vector_width != 8:
            raise ValueError(f"Sunway G1 requires SIMD width 8, got {vector_width}")
        if tile_n % vector_width:
            raise ValueError(
                f"Sunway GEMM tile N {tile_n} is not divisible by vector width {vector_width}"
            )

        active_cpes = 1
        if config.gemm_ownership == "mesh_2d":
            active_cpes = min(block_tiles_m, config.cpe_rows) * min(
                block_tiles_n, config.cpe_cols
            )

        argument_count = sum(param in func.buffer_map for param in func.params)
        element_bytes = 4
        ldm_bytes = (
            (tile_m * tile_k + tile_k * tile_n + tile_m * tile_n) * element_bytes
            + argument_count * _SW64_POINTER_BYTES
            + _REPLY_BYTES
        )
        if ldm_bytes > config.ldm_bytes_per_cpe:
            raise ValueError(
                f"Sunway G0 LDM plan uses {ldm_bytes} bytes, but target limit is "
                f"{config.ldm_bytes_per_cpe}"
            )

        return cls(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            block_tiles_m=block_tiles_m,
            block_tiles_n=block_tiles_n,
            k_panels=k_panels,
            global_m=block_tiles_m * tile_m,
            global_n=block_tiles_n * tile_n,
            global_k=k_panels * tile_k,
            workers=workers,
            cpe_rows=config.cpe_rows,
            cpe_cols=config.cpe_cols,
            active_cpes=active_cpes,
            ownership=config.gemm_ownership,
            compute=config.gemm_compute,
            vector_width=vector_width,
            stages=stages,
            input_dtype=input_dtype,
            accum_dtype=accum_dtype,
            ldm_bytes=ldm_bytes,
        )
