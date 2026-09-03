"""Register Sunway GEMM lowering implementations."""

from tilelang.sunway.target import is_sunway_target
from tilelang.tileop.gemm.registry import register_gemm_impl

from .gemm_scalar import GEMM_INST_SCALAR, GemmScalar
from .plan import SunwayGemmPlan


__all__ = ["SunwayGemmPlan"]


register_gemm_impl(
    "sunway.scalar",
    GEMM_INST_SCALAR,
    is_sunway_target,
    GemmScalar,
)
