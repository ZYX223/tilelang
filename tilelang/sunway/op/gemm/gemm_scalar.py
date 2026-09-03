"""Scalar correctness lowering for a single Sunway GEMM tile."""

from __future__ import annotations

from tilelang import language as T
from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.transform.simplify import _Simplify
from tvm import tirx
from tvm.ir import Range
from tvm.target import Target


GEMM_INST_SCALAR = "sunway.scalar"


class GemmScalar(GemmBase):
    """Expand one GEMM tile to scalar TIR owned by logical CPE zero."""

    def infer_layout(self, target: Target, thread_nums: int):
        return {}

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_index: tirx.PrimExpr,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        del layout_map, target, thread_bounds, mbar_phase_expr

        m, n, k = self.M, self.N, self.K
        a_buffer = self.ARegion.buffer
        b_buffer = self.BRegion.buffer
        c_buffer = self.CRegion.buffer
        trans_a = self.trans_A
        trans_b = self.trans_B
        clear_accum = self.clear_accum
        accum_dtype = self.accum_dtype

        # BufferRegion minima preserve sliced operands such as A[row, k].
        a0, a1 = (region.min for region in self.ARegion.region)
        b0, b1 = (region.min for region in self.BRegion.region)
        c0, c1 = (region.min for region in self.CRegion.region)

        @T.prim_func
        def _gemm_scalar() -> None:
            if thread_index == 0:
                if clear_accum:
                    for i, j in T.grid(m, n):
                        c_buffer[c0 + i, c1 + j] = T.cast(0, accum_dtype)
                for i, j, ki in T.grid(m, n, k):
                    a_i = ki if trans_a else i
                    a_j = i if trans_a else ki
                    b_i = j if trans_b else ki
                    b_j = ki if trans_b else j
                    c_buffer[c0 + i, c1 + j] += T.cast(
                        a_buffer[a0 + a_i, a1 + a_j]
                        * b_buffer[b0 + b_i, b1 + b_j],
                        accum_dtype,
                    )

        return _Simplify(_gemm_scalar, inline_let=True)
