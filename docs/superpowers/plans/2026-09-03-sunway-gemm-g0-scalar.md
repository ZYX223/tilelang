# Sunway GEMM G0 Scalar Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile canonical square and non-square tiled TileLang GEMM frontends through generic TileOp dispatch into scalar, single-active-CPE SW9A AOT kernels and prove numerical correctness on the real target.

**Architecture:** Keep the official `T.Kernel`/shared/fragment/pipeline/copy/GEMM program unchanged. Add a native Sunway TileOp target, lower the G0 program through `LowerTileOp`, serialize logical block tiles on the CPE side, and use CPE 0 as a correctness baseline with row-wise DMA and scalar accumulation. S1/S2/S3 remain TIR checkpoints; the C emitter only prints S3.

**Tech Stack:** Python 3.10, TileLang 0.1.14 source tree, TVM TIRx, C++ TileOp registries, pytest, CMake/Ninja, SWGCC-1307, MPE/CPE C, `swrun`

**Spec:** `docs/superpowers/specs/2026-09-03-sunway-gemm-backend-design.md`

## Global Constraints

- Use the existing TileLang language surface: `T.Kernel`, `T.alloc_shared`, `T.alloc_fragment`, `T.clear`, `T.Pipelined`, `T.copy`, and `T.gemm`.
- Do not add a public `SW.gemm`, Sunway-only allocation function, or whole-matrix `T.gemm(A_global, B_global, C_global)` shortcut.
- G0 supports static two-dimensional FP32 dense buffers, one CG, exactly `cpe_rows * cpe_cols` logical workers, and `num_stages=1`.
- G0 activates only CPE 0. Multi-CPE ownership, SIMD, double buffering, packed weights, tuning, and PyTorch integration belong to later milestones.
- TIR is authoritative. Python dataclasses hold analysis results only; generated C strings never determine scheduling.
- S1 retains official TileOps and thread bindings. S2 contains serial block-tile loops, CPE ownership, explicit LDM accounting, and abstract DMA/reply operations. S3 contains native SW9A calls and statically shaped buffers; the C emitter computes row-major flat-pointer offsets without changing manifest shapes.
- Keep C codegen mechanical and independent of the kernel name or GEMM shape.
- Preserve all current copy and runtime behavior. Existing dirty-worktree changes must not be reset or folded invisibly into GEMM commits.
- Generate and test on Dell `zyx@10.10.10.24` in `/mnt/sda/zyx/project/tilelang-paper1-clean` using `/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3`.
- Compile target artifacts with `/mnt/sda/zyx/toolchains/sw9a-sdk-overlay/bin/swgcc1307` and `/mnt/sda/zyx/toolchains/swgcc710-tools-SEA-1307`.
- Real acceptance requires a successful run on `root@10.10.10.22` through `swrun -E 64 -i`; generated C and host-side pytest are checkpoints only.

## File Structure

```text
src/sunway/
├── CMakeLists.txt             # Always-built Sunway target and TileOp sources
├── target_utils.h             # C++ TargetIsSunway declaration
├── target_utils.cc            # Detect the normalized target key
└── op/
    ├── copy.cc                # Canonical tiled-copy to abstract 2D DMA
    ├── fill.cc                # Thread-0 scalar fill for G0
    └── gemm.cc                # C++ GEMM instruction-key selection

tilelang/sunway/
├── __init__.py                # Import target, backend, and op registrations
├── pipeline.py                # S1/S2/S3 orchestration and common passes
├── transform.py               # Shared semantic-to-native leaves and dispatch
├── gemm_transform.py          # G0 worker/block/DMA materialization and verifiers
└── op/gemm/
    ├── __init__.py            # Register sunway.scalar
    ├── plan.py                # Whole-function SunwayScalarGemmPlan analysis
    └── gemm_scalar.py         # GemmBase implementation that emits scalar TIR

testing/python/sunway/
├── gemm_cases.py              # Reusable canonical GEMM PrimFunc factories
├── test_target.py             # Native target-routing tests
├── test_gemm_dispatch.py      # GEMM/Copy/Fill TileOp dispatch tests
├── test_gemm_transform.py     # S1/S2/S3 structural and verifier tests
└── test_backend_registration.py # AOT source/manifest integration tests

examples/sunway/
├── gemm_32.py                 # Generate the canonical G0 AOT project
├── gemm_32_main.c             # MPE reference and numerical check
├── gemm_m32_n16_k32.py        # Generate the non-square G0 AOT project
├── gemm_m32_n16_k32_main.c    # Non-square MPE reference and check
├── package_aot.py             # Compile any standalone generated project
└── run_aot.py                 # Deploy and run any standalone package
```

The existing flat `tilelang/sunway/analysis.py` remains the copy planner. Do not convert it into a package during G0; the GEMM plan lives with the GEMM implementation to avoid unrelated file churn.

---

### Task 0: Checkpoint The Verified Copy Baseline

**Files:**
- Review and stage only the pre-existing Sunway copy/runtime changes shown by `git status` before GEMM implementation starts.
- Do not modify source in this task.

**Interfaces:**
- Consumes: the currently verified `T.copy -> S1/S2/S3 -> MPE/CPE C` worktree state.
- Produces: a clean tracked baseline on which GEMM commits can modify `pipeline.py`, `transform.py`, and `codegen.py` without mixing histories.

- [ ] **Step 1: Record the current status and patch**

Run:

```bash
git status --short --branch
git diff --stat
git diff --check
```

Expected: the known Sunway copy/runtime files are modified or untracked; the two GEMM design commits are already in `HEAD`; `git diff --check` emits no errors.

- [ ] **Step 2: Run the complete existing Sunway suite on Dell**

Run on Dell:

```bash
cd /mnt/sda/zyx/project/tilelang-paper1-clean
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m pytest testing/python/sunway -q
```

Expected: `32 passed` before any GEMM implementation change.

- [ ] **Step 3: Review the exact baseline files before staging**

Run:

```bash
git diff -- docs/get_started/targets.md docs/index.md examples/sunway testing/python/sunway tilelang/sunway
git status --short -- docs/architecture docs/get_started/sunway.md docs/superpowers/plans
```

Expected: every change belongs to the previously completed Sunway copy/AOT/runtime work. Stop execution and ask the user about any unrelated file instead of staging it.

- [ ] **Step 4: Commit the verified baseline separately**

Stage the reviewed paths only:

```bash
git add docs/get_started/targets.md docs/index.md docs/architecture docs/get_started/sunway.md docs/superpowers/plans examples/sunway testing/python/sunway tilelang/sunway
git diff --cached --check
git commit -m "feat(sunway): checkpoint verified AOT copy backend"
```

Expected: one baseline commit; no unrelated path is staged; subsequent GEMM commits contain only their listed files.

---

### Task 1: Native Sunway Target Classification

**Files:**
- Create: `src/sunway/CMakeLists.txt`
- Create: `src/sunway/target_utils.h`
- Create: `src/sunway/target_utils.cc`
- Modify: `CMakeLists.txt`
- Modify: `src/backend/common/target_utils.h`
- Modify: `src/cpu/target_utils.cc`
- Create: `testing/python/sunway/test_target.py`

**Interfaces:**
- Consumes: normalized TVM target `kind=c`, key `sunway`, encoded config tag.
- Produces: C++ `bool TargetIsSunway(Target target)` and FFI function `tl.TargetIsSunway`; `TargetIsCPU` returns false for a Sunway carrier target.

- [ ] **Step 1: Write failing target-routing tests**

Add:

```python
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
```

- [ ] **Step 2: Run the test and observe the missing native predicate**

Run after syncing to Dell:

```bash
cmake --build build -j2
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m pytest testing/python/sunway/test_target.py -q
```

Expected: FAIL because `TargetIsSunway` is not registered and the normalized `c` carrier still satisfies `TargetIsCPU`.

- [ ] **Step 3: Add the always-built Sunway source group**

`src/sunway/CMakeLists.txt`:

```cmake
file(GLOB TILE_LANG_SUNWAY_SRCS
  src/sunway/op/*.cc
  src/sunway/target_utils.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_SUNWAY_SRCS})
```

Include it from the top-level `CMakeLists.txt` alongside the other backend-local source lists:

```cmake
include("${CMAKE_CURRENT_SOURCE_DIR}/src/sunway/CMakeLists.txt")
```

- [ ] **Step 4: Implement the C++ target predicate and CPU exclusion**

Declare `TargetIsSunway` in `src/sunway/target_utils.h`. Implement it by checking `target->keys` for the exact string `sunway`, and register `tl.TargetIsSunway` through `TVM_FFI_STATIC_INIT_BLOCK`.

Update `TargetIsCPU` so the normalized carrier cannot match the CPU TileOp implementations:

```cpp
bool TargetIsCPU(Target target) {
  if (target->GetTargetDeviceType() != kDLCPU) {
    return false;
  }
  for (const String &key : target->keys) {
    if (key == "sunway") {
      return false;
    }
  }
  return true;
}
```

Add `#include "sunway/target_utils.h"` to `src/backend/common/target_utils.h`.

- [ ] **Step 5: Rebuild and run target tests**

Run on Dell:

```bash
cmake --build build -j2
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m pytest testing/python/sunway/test_target.py testing/python/sunway/test_backend_registration.py::test_sunway_target_selects_dedicated_backend -q
```

Expected: all tests pass and no optional CUDA/ROCm SDK becomes a dependency of the Sunway sources.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt src/backend/common/target_utils.h src/cpu/target_utils.cc src/sunway/CMakeLists.txt src/sunway/target_utils.h src/sunway/target_utils.cc testing/python/sunway/test_target.py
git commit -m "feat(sunway): add native target classification"
```

---

### Task 2: Register The Sunway Scalar GEMM Implementation

**Files:**
- Create: `src/sunway/op/gemm.cc`
- Create: `tilelang/sunway/op/__init__.py`
- Create: `tilelang/sunway/op/gemm/__init__.py`
- Create: `tilelang/sunway/op/gemm/gemm_scalar.py`
- Modify: `tilelang/sunway/__init__.py`
- Create: `testing/python/sunway/gemm_cases.py`
- Create: `testing/python/sunway/test_gemm_dispatch.py`

**Interfaces:**
- Consumes: generic `tl.tileop.gemm`, `GemmBase`, C++ `RegisterGemmImpl`, and a Sunway target.
- Produces: instruction key `sunway.scalar` and class `GemmScalar`.

- [ ] **Step 1: Add the canonical reusable frontend fixture**

`testing/python/sunway/gemm_cases.py` defines a source-visible factory so the eager parser can inspect it:

```python
import tilelang.language as T


def make_gemm_32(*, workers: int = 64, num_stages: int = 1):
    @T.prim_func
    def gemm_32(
        A: T.Tensor((32, 32), "float32"),
        B: T.Tensor((32, 32), "float32"),
        C: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel(2, 2, threads=workers) as (bx, by):
            A_shared = T.alloc_shared((16, 32), "float32")
            B_shared = T.alloc_shared((32, 16), "float32")
            C_local = T.alloc_fragment((16, 16), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(1, num_stages=num_stages):
                T.copy(A[by * 16, ko * 32], A_shared)
                T.copy(B[ko * 32, bx * 16], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * 16, bx * 16])
    return gemm_32
```

In the same file, add `make_gemm_m32_n16_k32()` with `A=(32, 32)`,
`B=(32, 16)`, `C=(32, 16)`, and `T.Kernel(1, 2, threads=workers)`. It uses the
same `16 x 32`, `32 x 16`, and `16 x 16` local tiles and a single K panel. This
case proves that output-grid and manifest handling are not accidentally square.

- [ ] **Step 2: Write the failing dispatch test**

Add tests that assert:

```python
from tilelang.backend.module import create_backend_context
from tilelang.sunway.op.gemm.gemm_scalar import GemmScalar
from tilelang.tileop.gemm.registry import resolve_gemm_impl


def test_sunway_scalar_gemm_is_registered() -> None:
    target = create_backend_context({"kind": "sunway"}).target
    assert resolve_gemm_impl("sunway.scalar", target) is GemmScalar
```

Run the same `LayoutInference`/instruction-resolution check on
`make_gemm_m32_n16_k32()` so both public frontend shapes use the same generic
dispatch path.

- [ ] **Step 3: Run tests and observe missing modules/registration**

Run:

```bash
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m pytest testing/python/sunway/test_gemm_dispatch.py -q
```

Expected: FAIL because the Sunway GEMM implementation and plan do not exist.

- [ ] **Step 4: Register `sunway.scalar` in C++**

Model `src/sunway/op/gemm.cc` on the CPU target implementation:

```cpp
constexpr const char *kSunwayScalar = "sunway.scalar";

struct Gemm {
  static String SelectInst(const GemmNode &, int, Target) {
    return kSunwayScalar;
  }

  static std::pair<int, int> ComputeWarpPartition(
      const GemmWarpPolicyNode &policy, int, int, int, Target, String) {
    policy.m_warp = 1;
    policy.n_warp = 1;
    return {1, 1};
  }

  static bool ReuseExistingSharedLayout(String) { return false; }
};
```

Register it with `RegisterGemmImpl` and `MatchSunwayGemmTarget` calling `TargetIsSunway`.

- [ ] **Step 5: Implement `GemmScalar` as TIR, not C text**

Subclass `GemmBase`, return no inferred layout for G0, and emit a full tile only on logical worker zero:

```python
class GemmScalar(GemmBase):
    def infer_layout(self, target, thread_nums):
        return {}

    def lower(self, layout_map, target, thread_bounds, thread_index, mbar_phase_expr=None):
        M, N, K = self.M, self.N, self.K
        A = self.ARegion.buffer
        B = self.BRegion.buffer
        C = self.CRegion.buffer
        accum_dtype = self.accum_dtype
        clear_accum = self.clear_accum

        @T.prim_func
        def _gemm_scalar() -> None:
            if thread_index == 0:
                if clear_accum:
                    for i, j in T.grid(M, N):
                        C[i, j] = T.cast(0, accum_dtype)
                for i, j, k in T.grid(M, N, K):
                    C[i, j] += T.cast(A[i, k] * B[k, j], accum_dtype)

        return _Simplify(_gemm_scalar, inline_let=True)
```

The complete lowerer captures `a0/a1`, `b0/b1`, and `c0/c1` from the three
`BufferRegion` minima, plus `trans_A = self.trans_A` and
`trans_B = self.trans_B`. It computes indices exactly as follows:

```python
a_i = k if trans_A else i
a_j = i if trans_A else k
b_i = j if trans_B else k
b_j = k if trans_B else j
C[c0 + i, c1 + j] += T.cast(
    A[a0 + a_i, a1 + a_j] * B[b0 + b_i, b1 + b_j],
    accum_dtype,
)
```

- [ ] **Step 6: Import and register the implementation**

Register with:

```python
register_gemm_impl(
    "sunway.scalar",
    GEMM_INST_SCALAR,
    is_sunway_target,
    GemmScalar,
)
```

Import `tilelang.sunway.op` from `tilelang/sunway/__init__.py` after target normalization and before pipeline use.

- [ ] **Step 7: Rebuild and run the dispatch tests**

Run on Dell:

```bash
cmake --build build -j2
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m pytest testing/python/sunway/test_gemm_dispatch.py -q
```

Expected: the Sunway instruction key resolves to `GemmScalar` and all dispatch tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/sunway/op/gemm.cc tilelang/sunway/__init__.py tilelang/sunway/op testing/python/sunway/gemm_cases.py testing/python/sunway/test_gemm_dispatch.py
git commit -m "feat(sunway): register scalar GEMM TileOp"
```

---

### Task 3: Lower Canonical Copy And Fill TileOps

**Files:**
- Create: `src/sunway/op/copy.cc`
- Create: `src/sunway/op/fill.cc`
- Modify: `testing/python/sunway/test_gemm_dispatch.py`

**Interfaces:**
- Consumes: canonical global-to-shared A/B copies, fragment clear, and fragment-to-global C copy.
- Produces: abstract calls `tilelang_sunway_dma_get_2d` and `tilelang_sunway_dma_put_2d`, plus scalar FP32 fill loops, all guarded by logical worker zero.

- [ ] **Step 1: Write a failing full-frontend `LowerTileOp` test**

Bind `make_gemm_32()` to the Sunway target, run `LayoutInference` and `LowerTileOp`, then collect call names. Assert:

```python
assert "tl.tileop.copy" not in names
assert "tl.tileop.fill" not in names
assert "tl.tileop.gemm" not in names
assert names.count("tilelang_sunway_dma_get_2d") == 2
assert names.count("tilelang_sunway_dma_put_2d") == 1
```

Also assert that scalar multiply/add `BufferLoad` and `BufferStore` nodes remain, proving GEMM became TIR rather than a C template call.

- [ ] **Step 2: Run the test and observe missing Copy/Fill implementations**

Run:

```bash
cmake --build build -j2
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m pytest testing/python/sunway/test_gemm_dispatch.py::test_canonical_gemm_lowers_all_tileops -q
```

Expected: FAIL from target-specific Copy or Fill resolution.

- [ ] **Step 3: Implement G0 2D-copy descriptors**

Register a Sunway `CopyImpl` with empty layout inference. Its lowerer accepts exactly rank-2, same-dtype copies with compact innermost dimensions and these direction pairs:

```text
global -> shared/shared.dyn/local.fragment : tilelang_sunway_dma_get_2d
shared/shared.dyn/local.fragment -> global : tilelang_sunway_dma_put_2d
```

The semantic call arguments are:

```text
source_access_ptr
destination_access_ptr
rows
row_bytes
source_stride_bytes
destination_stride_bytes
```

Build the access pointers with `MakeAccessPtrFromRegion`, derive row strides from compact buffer shapes, and wrap the call in `thread_index == 0`. Reject local-to-local and non-compact innermost ranges with an `ICHECK` message naming the source/destination scopes.

- [ ] **Step 4: Implement the G0 fill lowerer**

Register a Sunway `FillImpl`. Build serial loops over `FillNode.region`, store `op.value` into `op.dst`, and wrap the full loop nest in `thread_index == 0`. Accept only `local.fragment`, `local`, `shared`, and `shared.dyn` destinations in G0.

- [ ] **Step 5: Rebuild and run dispatch tests**

Run:

```bash
cmake --build build -j2
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m pytest testing/python/sunway/test_gemm_dispatch.py -q
```

Expected: all TileOps lower, exactly two abstract gets and one abstract put remain, and no native `athread_*` call appears yet.

- [ ] **Step 6: Commit**

```bash
git add src/sunway/op/copy.cc src/sunway/op/fill.cc testing/python/sunway/test_gemm_dispatch.py
git commit -m "feat(sunway): lower tiled copy and fill operations"
```

---

### Task 4: Materialize And Verify G0 S2 TIR

**Files:**
- Create: `tilelang/sunway/op/gemm/plan.py`
- Create: `tilelang/sunway/gemm_transform.py`
- Modify: `tilelang/sunway/pipeline.py`
- Modify: `tilelang/sunway/transform.py`
- Create: `testing/python/sunway/test_gemm_transform.py`

**Interfaces:**
- Consumes: S1 canonical tiled TIR and the TileOp-lowered abstract 2D DMA descriptors.
- Produces: `SunwayScalarGemmPlan.from_prim_func(func, config)`, `lower_gemm_program_to_semantic_tir(mod, config)`, `verify_gemm_semantic_tir(mod, config)`, and S2 TIR with no TileOps or thread-binding loops.

- [ ] **Step 1: Write failing S2 boundary tests**

For `make_gemm_32()`, assert after S2:

```python
assert func.attrs["sunway.phase"] == "S2"
assert func.attrs["sunway.kernel_kind"] == "gemm_scalar"
assert int(func.attrs["sunway.cpe_count"]) == 64
assert int(func.attrs["sunway.active_cpes"]) == 1
assert int(func.attrs["sunway.block_tiles_m"]) == 2
assert int(func.attrs["sunway.block_tiles_n"]) == 2
assert int(func.attrs["sunway.tile_m"]) == 16
assert int(func.attrs["sunway.tile_n"]) == 16
assert int(func.attrs["sunway.tile_k"]) == 32
assert int(func.attrs["sunway.pipeline_stages"]) == 1
assert int(func.attrs["sunway.ldm_bytes"]) == (16 * 32 + 32 * 16 + 16 * 16) * 4 + 3 * 8 + 4
```

Also assert:

- block bindings became two serial loops;
- `threadIdx.x/y/z` bindings are gone;
- `tilelang_sunway_pe_id` occurs once;
- two DMA-get sites, one DMA-put site, and three reply-wait sites exist;
- every DMA site is nested in `_MYID == 0` ownership;
- no `athread_*` call or TileOp remains.

For `make_gemm_m32_n16_k32()`, assert `block_tiles_m == 2`,
`block_tiles_n == 1`, and the same tile and LDM attributes. Its S2 must still
contain two DMA-get sites and one DMA-put site.

`SunwayScalarGemmPlan` is a frozen dataclass with `tile_m`, `tile_n`, `tile_k`,
`workers`, `stages`, `input_dtype`, `accum_dtype`, and `ldm_bytes` fields.
`from_prim_func` derives them from the original allocations, GEMM
metadata, thread extent, pipeline annotation, and argument count. It rejects
symbolic dimensions, non-FP32 data, a worker count other than the configured
CPE count, a stage count other than one, and an LDM total above the target
budget. The total includes full A/B/C tiles, three SW64 pointers, and one
four-byte reply counter.

- [ ] **Step 2: Write failing legality tests**

Cover exact failures:

```python
with pytest.raises(ValueError, match="G0 requires 64 logical workers"):
    lower(make_gemm_32(workers=128))

with pytest.raises(ValueError, match="G0 requires num_stages=1"):
    lower(make_gemm_32(num_stages=2))

with pytest.raises(ValueError, match="LDM plan uses .* target limit"):
    lower(make_gemm_32(), ldm_bytes_per_cpe=2048)
```

Add mutated-IR tests for a missing wait, native `athread_get` in S2, residual `T.gemm`, and a non-serial block loop.

- [ ] **Step 3: Run tests and observe missing G0 transform APIs**

Run:

```bash
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m pytest testing/python/sunway/test_gemm_transform.py -q
```

Expected: FAIL because `gemm_transform.py` and pipeline dispatch do not exist.

- [ ] **Step 4: Implement official TileOp lowering order**

`lower_gemm_program_to_semantic_tir` applies:

```python
plan = SunwayScalarGemmPlan.from_prim_func(only_prim_func(mod), config)
mod = tilelang.transform.LayoutInference()(mod)
mod = tilelang.transform.LowerTileOp()(mod)
mod = lower_sunway_kernel_bindings(mod, config)
mod = expand_sunway_dma_2d(mod, config)
mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
mod = tilelang.transform.LowerOpaqueBlock()(mod)
mod = tirx.transform.Simplify()(mod)
mod = attach_scalar_gemm_plan(mod, plan)
```

Compute the plan before `LowerTileOp` removes `tl.tileop.gemm` and before
allocation-lowering changes the source scopes. Do not run `FlattenBuffer` in
S2; keep logical buffer shapes inspectable through codegen and manifest
creation.

- [ ] **Step 5: Lower kernel bindings without CUDA runtime semantics**

Implement a TIR mutator with these exact rules:

```text
blockIdx.x -> serial loop with the same min and extent
blockIdx.y -> serial loop with the same min and extent
blockIdx.z -> serial loop, extent must be one in G0
threadIdx.x -> remove loop and substitute loop var with semantic CPE id
threadIdx.y/z -> remove loop, require extent one, substitute zero
```

Insert one `Bind(pe_id, call_extern("tilelang_sunway_pe_id"))`. Require the original `threadIdx.x` extent to equal the configured CPE count. The scalar GEMM, fill, and DMA descriptors already carry `thread_index == 0`, so the resulting S2 activates only CPE 0.

- [ ] **Step 6: Expand abstract 2D DMA descriptors**

For each `tilelang_sunway_dma_{get,put}_2d`, allocate one function-level `int32 reply[1]` and replace the descriptor by:

```text
for row in serial(rows):
    reply[0] = 0
    tilelang_sunway_dma_get_or_put(
        src + row * src_stride_bytes / element_bytes,
        dst + row * dst_stride_bytes / element_bytes,
        row_bytes,
        &reply)
    tilelang_sunway_dma_wait(&reply, 1)
```

Require row bytes and both byte strides to be compile-time integers divisible by the element size. Require each row transfer to satisfy `config.dma_alignment`.

- [ ] **Step 7: Implement the GEMM S2 verifier**

The verifier checks the attributes above, exact CPE count, one active CPE, static serial block loops, no TileOps, no native calls, LDM budget, DMA alignment, and issue/wait pairing in each row-transfer sequence. It accepts multiple syntactic DMA sites inside loops; it does not compare a fixed dynamic execution count.

Dispatch `verify_semantic_tir` by `sunway.kernel_kind` so existing copy verification remains unchanged.

- [ ] **Step 8: Wire the GEMM path into the pipeline**

In `SunwayPassPipelineBody`, inspect S1 call names:

```python
if contains_tile_op(s1, "tl.tileop.gemm"):
    s2 = lower_gemm_program_to_semantic_tir(s1, config)
else:
    s2 = lower_tile_copy_to_semantic_tir(s1, config)
```

Reject a PrimFunc containing more than one GEMM site in G0. Keep the existing dump filenames `s1_annotated_tir.txt`, `s2_semantic_tir.txt`, and `s3_lowered_tir.txt`.

- [ ] **Step 9: Run S2 and copy regression tests**

Run:

```bash
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m pytest testing/python/sunway/test_gemm_transform.py testing/python/sunway/test_transform.py -q
```

Expected: all new S2 tests and existing copy tests pass.

- [ ] **Step 10: Commit**

```bash
git add tilelang/sunway/op/gemm/plan.py tilelang/sunway/gemm_transform.py tilelang/sunway/pipeline.py tilelang/sunway/transform.py testing/python/sunway/test_gemm_transform.py
git commit -m "feat(sunway): materialize scalar GEMM semantic TIR"
```

---

### Task 5: Lower G0 S2 To Codegen-Ready S3

**Files:**
- Modify: `tilelang/sunway/gemm_transform.py`
- Modify: `tilelang/sunway/transform.py`
- Modify: `tilelang/sunway/pipeline.py`
- Modify: `testing/python/sunway/test_gemm_transform.py`

**Interfaces:**
- Consumes: verified G0 S2 TIR.
- Produces: `lower_gemm_semantic_to_native_tir(mod, config)` and `verify_gemm_native_tir(mod, config)` with native DMA calls and statically shaped compact buffers.

- [ ] **Step 1: Write failing S3 tests**

Assert:

```python
assert func.attrs["sunway.phase"] == "S3"
assert "tilelang_sunway_dma_get" not in names
assert "tilelang_sunway_dma_put" not in names
assert "tilelang_sunway_dma_wait" not in names
assert names.count("athread_get") == 2
assert names.count("athread_put") == 1
assert names.count("tilelang_sunway_reply_wait") == 3
assert all(all(isinstance(dim, tirx.IntImm) for dim in buffer.shape) for buffer in allocated_buffers(func))
```

Call counts are static call sites, not dynamic row iterations. Add rejection
tests for a residual semantic call, a residual thread binding, a symbolic buffer
shape, missing scalar multiply/add, and LDM metadata above the target budget.

- [ ] **Step 2: Run tests and observe S3 still uses copy-only verification**

Run:

```bash
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m pytest testing/python/sunway/test_gemm_transform.py -q
```

Expected: S3 tests fail because the verifier requires `kernel_kind=copy` or leaves unresolved GEMM semantic calls.

- [ ] **Step 3: Reuse shared semantic leaf lowering**

Extend the existing semantic map only with names actually present in S2. Preserve:

```python
"tilelang_sunway_pe_id" -> "_MYID"
"tilelang_sunway_dma_get" -> "athread_get"
"tilelang_sunway_dma_put" -> "athread_put"
"tilelang_sunway_dma_wait" -> "tilelang_sunway_reply_wait"
```

Do not introduce a native `gemm` extern call. Scalar multiplication and accumulation remain ordinary TIR.

- [ ] **Step 4: Apply codegen-preparation passes after semantic lowering**

After native leaf replacement, apply:

```python
mod = tilelang.transform.Simplify()(mod)
mod = tirx.transform.Simplify()(mod)
mod = tirx.transform.RemoveNoOp()(mod)
```

Then mark S3 and run `verify_gemm_native_tir`. The verifier enforces static
compact allocated and parameter buffers, native DMA/wait ordering, scalar FP32
arithmetic, no TileOps, and no thread bindings. Keeping parameter shapes intact
ensures the generated manifest records `(32, 32)` rather than `(1024,)`.

- [ ] **Step 5: Dispatch native verification by kernel kind**

Keep the copy verifier intact and route `gemm_scalar` to `verify_gemm_native_tir`. A missing or unknown `sunway.kernel_kind` is a compilation error.

- [ ] **Step 6: Run S3 and copy regression tests**

Run:

```bash
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m pytest testing/python/sunway/test_gemm_transform.py testing/python/sunway/test_transform.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add tilelang/sunway/gemm_transform.py tilelang/sunway/transform.py tilelang/sunway/pipeline.py testing/python/sunway/test_gemm_transform.py
git commit -m "feat(sunway): lower scalar GEMM to native TIR"
```

---

### Task 6: Extend Mechanical C Emission For Scalar GEMM

**Files:**
- Modify: `tilelang/sunway/codegen.py`
- Modify: `testing/python/sunway/test_backend_registration.py`

**Interfaces:**
- Consumes: verified G0 S3 with serial loops, statically shaped `BufferLoad`/`BufferStore`, scalar casts/arithmetic, conditions, and native DMA calls.
- Produces: MPE C, CPE C, common header, and manifest without a GEMM-specific source template.

- [ ] **Step 1: Write a failing AOT source test**

Lower `make_gemm_32()` with output index 2. Assert that generated CPE C contains:

```python
assert "void gemm_32_cpe(" in cpe
assert "_MYID == 0" in cpe
assert "for (int bx = 0; bx < 2; ++bx)" in cpe
assert "for (int by = 0; by < 2; ++by)" in cpe
assert "athread_get(PE_MODE" in cpe
assert "athread_put(PE_MODE" in cpe
assert "C_local[" in cpe
assert "A_shared[" in cpe
assert "B_shared[" in cpe
assert " * " in cpe
assert " + " in cpe
```

Assert the common header has three `float *` fields, MPE C has one spawn/join
pair and no `athread_init`, the manifest roles are `input`, `input`, `output`,
and every manifest shape is `(32, 32)`.

Lower `make_gemm_m32_n16_k32()` in a second test and assert its manifest shapes
are `(32, 32)`, `(32, 16)`, and `(32, 16)`, while the generated CPE source uses
one x block tile and two y block tiles. Both tests must pass through the same
generic `_CPEEmitter`; do not add a shape-name branch.

- [ ] **Step 2: Run the test and observe unsupported `BufferLoad`**

Run:

```bash
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m pytest testing/python/sunway/test_backend_registration.py::test_sunway_scalar_gemm_lowers_to_aot_project -q
```

Expected: FAIL from `_CPEEmitter.emit_expr` on `BufferLoad` or from existing
multi-index store emission against a flat C pointer.

- [ ] **Step 3: Add generic row-major buffer offsets**

Add a helper that calculates a row-major C offset from the static buffer shape:

```python
def _emit_flat_index(self, buffer: tirx.Buffer, indices: list[object]) -> str:
    if len(indices) != len(buffer.shape):
        raise TypeError("Sunway CPE codegen buffer rank mismatch")
    offset = "0"
    for index, dim in zip(indices, buffer.shape, strict=True):
        if not isinstance(dim, tirx.IntImm):
            raise TypeError("Sunway CPE codegen requires static buffer shapes")
        offset = f"(({offset}) * {int(dim)} + ({self.emit_expr(index)}))"
    return offset

if isinstance(expr, tirx.BufferLoad):
    base = self._buffer_base(expr.buffer)
    return f"{base}[{self._emit_flat_index(expr.buffer, list(expr.indices))}]"
```

Use `_emit_flat_index` for `BufferStore` as well. This is mechanical buffer-layout
emission and contains no GEMM-specific shape or loop logic.

- [ ] **Step 4: Run AOT and complete Sunway tests**

Run:

```bash
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m pytest testing/python/sunway -q
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m ruff check tilelang/sunway testing/python/sunway examples/sunway
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m compileall -q tilelang/sunway testing/python/sunway examples/sunway
```

Expected: all tests, Ruff, and compileall pass; existing copy source assertions remain unchanged.

- [ ] **Step 5: Commit**

```bash
git add tilelang/sunway/codegen.py testing/python/sunway/test_backend_registration.py
git commit -m "feat(sunway): emit scalar GEMM from lowered TIR"
```

---

### Task 7: Add A Reproducible Standalone GEMM Package

**Files:**
- Create: `examples/sunway/gemm_32.py`
- Create: `examples/sunway/gemm_32_main.c`
- Create: `examples/sunway/gemm_m32_n16_k32.py`
- Create: `examples/sunway/gemm_m32_n16_k32_main.c`
- Create: `examples/sunway/package_aot.py`
- Create: `examples/sunway/run_aot.py`
- Modify: `testing/python/sunway/test_runtime.py`

**Interfaces:**
- Consumes: generated `gemm_32` project, `SunwayLibraryGenerator`, `SunwayToolchain`, and `SunwaySSHExecutor`.
- Produces: a standalone `gemm_32` hybrid executable package and one-command SW9A deployment.

- [ ] **Step 1: Write failing example-contract tests**

Assert:

```python
def test_gemm_main_owns_crts_initialization_once() -> None:
    source = gemm_main_path().read_text()
    assert source.count("athread_init();") == 1
    assert "void gemm_32(float *A, float *B, float *C);" in source
    assert "gemm_32 passed: M=32 N=32 K=32" in source


def test_generic_aot_scripts_do_not_name_copy_kernel() -> None:
    package_source = package_aot_path().read_text()
    run_source = run_aot_path().read_text()
    assert "copy_128" not in package_source
    assert "copy_128" not in run_source
```

Add the same initialization-count, exported-signature, and success-line checks
for `gemm_m32_n16_k32_main.c`.

- [ ] **Step 2: Implement the generator**

`gemm_32.py` imports `make_gemm_32` or defines the identical source-visible PrimFunc, then calls:

```python
tilelang.lower(
    gemm_32,
    target={
        "kind": "sunway",
        "output_dir": str(output_dir),
        "output_indices": [2],
    },
    runtime_only=True,
)
```

Copy `gemm_32_main.c` into the generated directory after lowering.

`gemm_m32_n16_k32.py` follows the identical flow with
`make_gemm_m32_n16_k32`, output index 2, and its matching MPE main file. The
backend API and packaging code are shared; only the frontend shape and numerical
oracle constants differ.

- [ ] **Step 3: Implement the MPE numerical oracle**

Use static `float A[32 * 32]`, `B`, `C`, and `expected`. Initialize A and B
with bounded deterministic values, compute `expected[i*N+j]` in a host triple
loop, call `athread_init()` exactly once, call `gemm_32`, and reject an element
without requiring `libm`:

```c
float scale = expected[index] < 0.0f ? -expected[index] : expected[index];
float diff = C[index] - expected[index];
scale = scale < 1.0f ? 1.0f : scale;
diff = diff < 0.0f ? -diff : diff;
if (diff > 1.0e-4f * scale) {
    fprintf(stderr, "gemm mismatch at (%d,%d): got=%f expected=%f\n", i, j, C[index], expected[index]);
    return 1;
}
```

Print exactly `gemm_32 passed: M=32 N=32 K=32` on success.

The non-square oracle uses `M=32`, `N=16`, and `K=32`, applies the same
tolerance, and prints exactly `gemm_m32_n16_k32 passed: M=32 N=16 K=32`.

- [ ] **Step 4: Implement generic packaging and deployment CLIs**

`package_aot.py` reads `manifest.json`, builds `SunwayToolchain.from_sdk_roots`, and calls `SunwayLibraryGenerator.compile_executable()`.

`run_aot.py` accepts `--package-dir`, `--remote-host`, `--remote-root`, `--executable`, and optional `--deployment-id`, then calls `SunwaySSHExecutor.deploy_and_run`. It relays stdout/stderr and exits with the target return code.

- [ ] **Step 5: Run example-contract and complete host tests**

Run:

```bash
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m pytest testing/python/sunway -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add examples/sunway/gemm_32.py examples/sunway/gemm_32_main.c examples/sunway/gemm_m32_n16_k32.py examples/sunway/gemm_m32_n16_k32_main.c examples/sunway/package_aot.py examples/sunway/run_aot.py testing/python/sunway/test_runtime.py
git commit -m "test(sunway): add standalone scalar GEMM package"
```

---

### Task 8: Rebuild On Dell And Prove G0 On SW9A

**Files:**
- Modify: `docs/get_started/sunway.md`
- Generated only: `/tmp/tilelang-gemm-32-generated-20260903`
- Generated only: `/tmp/tilelang-gemm-32-package-20260903`
- Generated only: `/tmp/tilelang-gemm-m32-n16-k32-generated-20260903`
- Generated only: `/tmp/tilelang-gemm-m32-n16-k32-package-20260903`

**Interfaces:**
- Consumes: all G0 implementation tasks and the existing Dell-to-SW9A direct deployment path.
- Produces: fresh host regression evidence, matching SWGCC compile evidence, and a real `swrun` numerical success record.

- [ ] **Step 1: Sync the reviewed branch to Dell without overwriting remote-only artifacts**

Use the existing repository remote or `rsync` source files while excluding `build`, `.git`, Python caches, and generated packages. Confirm Dell `git rev-parse HEAD` matches the local implementation `HEAD` before rebuilding.

- [ ] **Step 2: Reconfigure and rebuild TileLang with Sunway always-built sources**

Run on Dell using the repository's current configure options, then:

```bash
cd /mnt/sda/zyx/project/tilelang-paper1-clean
cmake --build build -j2
```

Expected: `libtilelang.so` links `src/sunway/target_utils.cc` and the three Sunway TileOp source files without requiring a CUDA runtime on Dell.

- [ ] **Step 3: Run final host verification**

Run:

```bash
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m pytest testing/python/sunway -q
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m ruff check tilelang/sunway testing/python/sunway examples/sunway
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 -m compileall -q tilelang/sunway testing/python/sunway examples/sunway
git diff --check
```

Expected: the complete Sunway suite passes and all static checks succeed.

- [ ] **Step 4: Generate the canonical G0 project**

Run:

```bash
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 examples/sunway/gemm_32.py --output-dir /tmp/tilelang-gemm-32-generated-20260903
```

Inspect all seven artifacts: S1, S2, S3, MPE C, CPE C, the common header, and the manifest. Confirm S1 retains the official frontend, S2 contains semantic DMA and scalar arithmetic, and S3 contains only native leaves.

Generate the non-square project through the same path:

```bash
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 examples/sunway/gemm_m32_n16_k32.py \
  --output-dir /tmp/tilelang-gemm-m32-n16-k32-generated-20260903
```

Inspect the same seven artifacts and confirm its manifest shapes are
`(32, 32)`, `(32, 16)`, and `(32, 16)`.

- [ ] **Step 5: Cross-compile with the matching 1307 SDK**

Run:

```bash
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 examples/sunway/package_aot.py \
  --generated-dir /tmp/tilelang-gemm-32-generated-20260903 \
  --package-dir /tmp/tilelang-gemm-32-package-20260903 \
  --toolchain-root /mnt/sda/zyx/toolchains/swgcc710-tools-SEA-1307 \
  --overlay-root /mnt/sda/zyx/toolchains/sw9a-sdk-overlay
```

Expected: `/tmp/tilelang-gemm-32-package-20260903/gemm_32` exists and the compiler reports no implicit declaration, ABI, or mixed-link error.

Compile the non-square project:

```bash
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 examples/sunway/package_aot.py \
  --generated-dir /tmp/tilelang-gemm-m32-n16-k32-generated-20260903 \
  --package-dir /tmp/tilelang-gemm-m32-n16-k32-package-20260903 \
  --toolchain-root /mnt/sda/zyx/toolchains/swgcc710-tools-SEA-1307 \
  --overlay-root /mnt/sda/zyx/toolchains/sw9a-sdk-overlay
```

Expected: `/tmp/tilelang-gemm-m32-n16-k32-package-20260903/gemm_m32_n16_k32`
exists with the same clean compiler result.

- [ ] **Step 6: Deploy directly and run on SW9A**

Run:

```bash
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 examples/sunway/run_aot.py \
  --package-dir /tmp/tilelang-gemm-32-package-20260903 \
  --remote-host root@10.10.10.22 \
  --remote-root /tmp/tilelang-runs \
  --executable gemm_32 \
  --deployment-id gemm-g0-scalar-20260903
```

Expected target output:

```text
gemm_32 passed: M=32 N=32 K=32
```

Expected return code: 0.

Deploy the non-square package:

```bash
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 examples/sunway/run_aot.py \
  --package-dir /tmp/tilelang-gemm-m32-n16-k32-package-20260903 \
  --remote-host root@10.10.10.22 \
  --remote-root /tmp/tilelang-runs \
  --executable gemm_m32_n16_k32 \
  --deployment-id gemm-g0-nonsquare-20260903
```

Expected target output:

```text
gemm_m32_n16_k32 passed: M=32 N=16 K=32
```

Expected return code: 0.

- [ ] **Step 7: Record the reproducible G0 workflow**

Add a `Scalar GEMM G0` subsection to `docs/get_started/sunway.md` containing the exact generation, packaging, and deployment commands above. State explicitly that G0 uses one active CPE and is a correctness baseline, not a performance claim.

- [ ] **Step 8: Run the documentation diff check and commit**

```bash
git diff --check
git add docs/get_started/sunway.md
git commit -m "docs(sunway): record scalar GEMM target validation"
```

---

## G0 Completion Gate

G0 is complete only when all conditions hold:

- the canonical official TileLang tiled GEMM source is unchanged by the Sunway frontend;
- native target dispatch selects `sunway.scalar` and does not match CPU;
- S1/S2/S3 dumps satisfy their verifiers;
- codegen contains no GEMM template or kernel-name special case;
- every existing Sunway copy/runtime test still passes;
- SWGCC-1307 compiles and links the generated project;
- both square and non-square real SW9A runs print their exact success lines and exit zero;
- documentation states that only CPE 0 is active and makes no performance claim.

After this gate, write a separate G1 plan using the verified SWGCC SIMD probe results. G1 must first distribute output ownership across 64 CPEs, then add multi-K SIMD; G2 adds two-stage buffering, G3 adds row/column communication, and G4 adds offline tuning and PyTorch packaging.
