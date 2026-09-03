from __future__ import annotations

import pytest
import tilelang
from tilelang.backend.module import create_backend_context
from tilelang.sunway.target import get_sunway_target_config
from tvm.target import Target


def test_native_target_predicates_separate_sunway_from_cpu() -> None:
    target = create_backend_context({"kind": "sunway"}).target

    assert tilelang._ffi_api.TargetIsSunway(target)
    assert not tilelang._ffi_api.TargetIsCPU(target)


def test_plain_c_target_remains_cpu() -> None:
    target = Target("c")

    assert not tilelang._ffi_api.TargetIsSunway(target)
    assert tilelang._ffi_api.TargetIsCPU(target)


def test_sunway_gemm_schedule_config_round_trips_through_target() -> None:
    target = create_backend_context(
        {
            "kind": "sunway",
            "gemm_ownership": "mesh_2d",
            "gemm_compute": "simd",
        }
    ).target

    config = get_sunway_target_config(target)

    assert config.gemm_ownership == "mesh_2d"
    assert config.gemm_compute == "simd"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gemm_ownership", "modulo"),
        ("gemm_compute", "cuda"),
    ],
)
def test_sunway_rejects_unknown_gemm_schedule_values(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=rf"unsupported Sunway {field} {value!r}"):
        create_backend_context({"kind": "sunway", field: value})
