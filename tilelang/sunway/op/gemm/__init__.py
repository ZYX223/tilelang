"""Register Sunway GEMM lowering implementations."""

from tilelang.sunway.target import is_sunway_target
from tilelang.tileop.gemm.registry import register_gemm_impl

from .gemm_scalar import GEMM_INST_SCALAR, GemmScalar
from .gemm_vmad import lower_gemm_compute_to_simd
from .lower import lower_gemm_program_to_semantic_tir, lower_gemm_semantic_to_native_tir
from .plan import SunwayGemmPlan
from .verify import verify_gemm_native_tir, verify_gemm_semantic_tir


__all__ = [
    "SunwayGemmPlan",
    "lower_gemm_compute_to_simd",
    "lower_gemm_program_to_semantic_tir",
    "lower_gemm_semantic_to_native_tir",
    "verify_gemm_native_tir",
    "verify_gemm_semantic_tir",
]


register_gemm_impl(
    "sunway.scalar",
    GEMM_INST_SCALAR,
    is_sunway_target,
    GemmScalar,
)
