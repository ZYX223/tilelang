from __future__ import annotations

import tilelang
from tilelang.backend.module import create_backend_context
from tvm.target import Target


def test_native_target_predicates_separate_sunway_from_cpu() -> None:
    target = create_backend_context({"kind": "sunway"}).target

    assert tilelang._ffi_api.TargetIsSunway(target)
    assert not tilelang._ffi_api.TargetIsCPU(target)


def test_plain_c_target_remains_cpu() -> None:
    target = Target("c")

    assert not tilelang._ffi_api.TargetIsSunway(target)
    assert tilelang._ffi_api.TargetIsCPU(target)
