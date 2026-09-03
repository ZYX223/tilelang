# Sunway GEMM G1 Distributed Multi-K SIMD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distribute official TileLang GEMM output tiles over the SW9A 8x8 CPE mesh, accumulate multiple K panels, and execute the verified SWGCC-1307 FP32x8 SIMD path.

**Architecture:** Preserve the G0 frontend and scalar single-CPE schedule as an explicit oracle. Add backend schedule controls, materialize a two-dimensional mesh-stride ownership schedule in S2, lower an abstract FP32x8 FMA to a validated S3 native leaf, and keep final C emission independent of kernel names and matrix shapes.

**Tech Stack:** Python 3.10, TileLang, TVM TIR/TIRx, pytest, C++ TileOp registration, SWGCC-1307, MPE/CPE C, `athread`, `floatv8`, `simd_vmas`, Dell cross-compilation host, SW9A target through `swrun -E 64 -i`.

**Spec:** `docs/superpowers/specs/2026-09-03-sunway-gemm-g1-distributed-simd-design.md`

## Global Constraints

- Keep the official TileLang frontend unchanged; add no Sunway-only language call.
- Defaults remain `gemm_ownership="single"` and `gemm_compute="scalar"` so G0 stays reproducible.
- G1 supports static, no-tail FP32 GEMM only: M and N are divisible by 16, K is divisible by 32.
- G1 uses synchronous DMA and one A buffer, one B buffer, and one C accumulator per CPE.
- S2 contains semantic ownership, DMA, and compute leaves; S3 contains only validated native leaves.
- Do not add double buffering, mesh communication, packed weights, PyTorch packaging, or a tuning database.
- Use Dell `/mnt/sda/zyx/project/tilelang-paper1-clean` for TileLang/TVM tests and SWGCC-1307 compilation.
- Use `root@10.10.10.22` only for packaged target probes and numerical runs.
- Every target success claim requires a fresh zero return code and captured success line.

---

### Task 1: Add Stable GEMM Schedule Controls

**Files:**
- Modify: `tilelang/sunway/target.py`
- Modify: `testing/python/sunway/test_target.py`

**Interfaces:**
- Consumes: `SunwayTargetConfig.from_mapping()` and target-tag round trip.
- Produces: `gemm_ownership: str` in `{single, mesh_2d}` and `gemm_compute: str` in `{scalar, simd}`.

- [ ] **Step 1: Write failing target round-trip and rejection tests**

Add tests that normalize this target and recover both fields:

```python
target = determine_target(
    {
        "kind": "sunway",
        "gemm_ownership": "mesh_2d",
        "gemm_compute": "simd",
    }
)
config = get_sunway_target_config(target)
assert config.gemm_ownership == "mesh_2d"
assert config.gemm_compute == "simd"
```

Also assert that `gemm_ownership="modulo"` and `gemm_compute="cuda"` raise
`ValueError` naming the invalid field and value.

- [ ] **Step 2: Run the focused tests and observe RED**

Run on Dell:

```bash
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 \
  -m pytest testing/python/sunway/test_target.py -q
```

Expected: failures because the recovered config lacks both fields and invalid
values are accepted.

- [ ] **Step 3: Implement immutable config fields and validation**

Add fields to `SunwayTargetConfig`, include them in `from_mapping()` and
`to_mapping()`, and validate exact values in `__post_init__()`:

```python
gemm_ownership: str = "single"
gemm_compute: str = "scalar"

def __post_init__(self) -> None:
    if self.gemm_ownership not in {"single", "mesh_2d"}:
        raise ValueError(f"unsupported Sunway gemm_ownership {self.gemm_ownership!r}")
    if self.gemm_compute not in {"scalar", "simd"}:
        raise ValueError(f"unsupported Sunway gemm_compute {self.gemm_compute!r}")
```

- [ ] **Step 4: Run focused and complete Sunway tests**

Expected: focused tests pass and the complete suite keeps all G0 tests green.

- [ ] **Step 5: Commit**

```bash
git add tilelang/sunway/target.py testing/python/sunway/test_target.py
git commit -m "feat(sunway): configure GEMM ownership and compute"
```

---

### Task 2: Generalize The Static GEMM Plan For G1

**Files:**
- Modify: `tilelang/sunway/op/gemm/plan.py`
- Modify: `tilelang/sunway/op/gemm/__init__.py`
- Modify: `testing/python/sunway/gemm_cases.py`
- Modify: `testing/python/sunway/test_gemm_transform.py`

**Interfaces:**
- Consumes: canonical `tl.tileop.gemm`, block bindings, pipeline annotation, and Task 1 config.
- Produces: `SunwayGemmPlan.from_prim_func(func, config)` with tile-grid, K-panel, mesh, ownership, compute, vector, active-CPE, and LDM metadata.

- [ ] **Step 1: Add a canonical `128x128x64` factory**

Add `make_gemm_128_k64()` using the same frontend pattern as G0:

```python
with T.Kernel(8, 8, threads=64) as (bx, by):
    A_shared = T.alloc_shared((16, 32), "float32")
    B_shared = T.alloc_shared((32, 16), "float32")
    C_local = T.alloc_fragment((16, 16), "float32")
    T.clear(C_local)
    for ko in T.Pipelined(2, num_stages=1):
        T.copy(A[by * 16, ko * 32], A_shared)
        T.copy(B[ko * 32, bx * 16], B_shared)
        T.gemm(A_shared, B_shared, C_local)
    T.copy(C_local, C[by * 16, bx * 16])
```

- [ ] **Step 2: Write failing plan tests**

For `mesh_2d/simd`, assert:

```python
assert plan.block_tiles_m == 8
assert plan.block_tiles_n == 8
assert plan.k_panels == 2
assert plan.global_m == 128
assert plan.global_n == 128
assert plan.global_k == 64
assert plan.cpe_rows == 8 and plan.cpe_cols == 8
assert plan.active_cpes == 64
assert plan.ownership == "mesh_2d"
assert plan.compute == "simd"
assert plan.vector_width == 8
```

Add failures for SIMD width not dividing tile N and a K-panel extent that is
not a positive static integer.

- [ ] **Step 3: Run tests and observe RED**

Run the plan-focused tests on Dell. Expected: missing factory/class fields.

- [ ] **Step 4: Rename and extend the plan**

Rename `SunwayScalarGemmPlan` to `SunwayGemmPlan`. Derive block extents from
`blockIdx.x/y`, K-panel count from the one annotated GEMM pipeline loop, and
global dimensions from tile sizes times extents. Compute:

```python
active_cpes = (
    min(block_tiles_m, config.cpe_rows)
    * min(block_tiles_n, config.cpe_cols)
    if config.gemm_ownership == "mesh_2d"
    else 1
)
```

Keep the existing LDM formula because G1 remains single-buffered. Reject SIMD
unless `tile_n % simd_width == 0` and `simd_width == 8`.

- [ ] **Step 5: Run focused and complete tests, then commit**

```bash
git add tilelang/sunway/op/gemm testing/python/sunway/gemm_cases.py \
  testing/python/sunway/test_gemm_transform.py
git commit -m "feat(sunway): plan distributed multi-K GEMM"
```

---

### Task 3: Materialize Two-Dimensional Mesh Ownership

**Files:**
- Modify: `tilelang/sunway/gemm_transform.py`
- Modify: `tilelang/sunway/codegen.py`
- Modify: `testing/python/sunway/test_gemm_transform.py`
- Modify: `testing/python/sunway/test_runtime.py`

**Interfaces:**
- Consumes: `SunwayGemmPlan` and the post-`LowerTileOp` block/thread loops.
- Produces: S2 mesh-stride loops and metadata with no `pe_id == 0` guard when ownership is `mesh_2d`.

- [ ] **Step 1: Write failing S2 ownership tests**

Lower `make_gemm_128_k64()` using `mesh_2d/scalar`. Assert three semantic
bindings (`pe_id`, `pe_row`, `pe_col`) and that block tile coordinates have the
equivalent schedule:

```text
bx = pe_col + bx_round * cpe_cols, guarded by bx < block_tiles_n
by = pe_row + by_round * cpe_rows, guarded by by < block_tiles_m
```

Assert `sunway.active_cpes == 64`, `sunway.ownership == "mesh_2d"`, no worker-zero
guard, and all three DMA sites are inside the owned tile body. Add a `10 by 9`
synthetic tile-grid structural test proving multiple mesh strides and guarded
last tiles.

- [ ] **Step 2: Run focused tests and observe RED**

Expected: current S2 reports one active CPE and contains `pe_id == 0`.

- [ ] **Step 3: Implement mesh-stride lowering**

Keep the current single-CPE branch unchanged. For `mesh_2d`:

1. bind `pe_id = tilelang_sunway_pe_id()`;
2. bind `pe_row = pe_id // cpe_cols` and `pe_col = pe_id % cpe_cols`;
3. replace the logical worker loop with its body;
4. rewrite each block coordinate as a static round loop plus a bound coordinate;
5. guard the coordinate against the original block extent.

The round-loop extents are `ceildiv(block_tiles_m, cpe_rows)` and
`ceildiv(block_tiles_n, cpe_cols)`. This keeps every TIR `For.extent`
non-negative and allows the existing structural C emitter to handle the loop.

- [ ] **Step 4: Strengthen S2/S3 verifiers**

For mesh ownership, require one PE-id semantic call, exact mesh metadata,
coordinate bindings derived from the PE id, positive static round extents,
configured row/column strides, guards for partial mesh rounds, and no
single-worker guard. Enumerate the static tile grid in the verifier and prove
that every tile has exactly one owner.

- [ ] **Step 5: Extend generic codegen only for newly surviving expressions**

Add expression support only if the S3 dump contains a standard TIR node the
emitter does not yet handle. Do not add checks for GEMM symbols or shapes.

- [ ] **Step 6: Verify generated C structure**

Assert generated C contains `_MYID / 8`, `_MYID % 8`, the row/column coordinate
guards, and no `_MYID == 0` for the G1 target. Keep the existing G0 source tests
unchanged and passing.

- [ ] **Step 7: Run all Sunway tests and commit**

```bash
git add tilelang/sunway/gemm_transform.py tilelang/sunway/codegen.py \
  testing/python/sunway/test_gemm_transform.py testing/python/sunway/test_runtime.py
git commit -m "feat(sunway): distribute GEMM tiles over CPE mesh"
```

---

### Task 4: Prove Distributed Multi-K Scalar Correctness

**Files:**
- Create: `examples/sunway/gemm_128_k64.py`
- Create: `examples/sunway/gemm_128_k64_main.c`
- Modify: `testing/python/sunway/test_runtime.py`

**Interfaces:**
- Consumes: Task 3 mesh S2/S3 and existing generic `package_aot.py`/`run_aot.py`.
- Produces: a deterministic `128x128x64` scalar AOT package and real SW9A checkpoint.

- [ ] **Step 1: Write failing source-contract tests**

Require the example to use only official frontend calls, accept
`--compute scalar|simd`, pass `gemm_ownership="mesh_2d"`, and copy its numerical
main into the generated directory. Require the main to initialize deterministic
binary-fraction inputs, compute a host triple-loop reference, call
`athread_init()` exactly once, and print:

```text
gemm_128_k64 passed: M=128 N=128 K=64
```

- [ ] **Step 2: Implement the example and numerical main**

The Python frontend remains one `gemm_128_k64` PrimFunc. CLI selection changes
only `target["gemm_compute"]`; it does not branch inside the DSL program.

- [ ] **Step 3: Generate and inspect S1/S2/S3 on Dell**

Use `/tmp/tilelang-gemm-g1-scalar-generated-20260903`. Confirm S1 contains two
K panels, S2 has mesh ownership and scalar multiply/add, and S3 has native DMA.

- [ ] **Step 4: Cross-compile and run on SW9A**

Package into `/tmp/tilelang-gemm-g1-scalar-package-20260903` with the existing
SWGCC-1307 roots. Deploy with id `gemm-g1-multik-scalar-20260903`. Require the
exact success line and return code zero.

- [ ] **Step 5: Commit only after target success**

```bash
git add examples/sunway/gemm_128_k64.py examples/sunway/gemm_128_k64_main.c \
  testing/python/sunway/test_runtime.py
git commit -m "test(sunway): prove distributed multi-K scalar GEMM"
```

---

### Task 5: Compile And Run The SWGCC SIMD Probe

**Files:**
- Create: `examples/sunway/probes/simd_vmas_f32x8_common.h`
- Create: `examples/sunway/probes/simd_vmas_f32x8_mpe.c`
- Create: `examples/sunway/probes/simd_vmas_f32x8_cpe.c`
- Create: `examples/sunway/probes/README.md`

**Interfaces:**
- Consumes: SWGCC-1307 `shared_include/simd.h` and existing toolchain command contract.
- Produces: target evidence for `floatv8`, `simd_set_floatv8`, and `simd_vmas`.

- [ ] **Step 1: Add the smallest mixed MPE/CPE numerical probe**

Only CPE 0 computes eight lanes. It DMA-loads B and C, broadcasts scalar A with
`simd_set_floatv8`, evaluates `simd_vmas(A, B, C)`, DMA-stores the result, and
prints this exact MPE success line after comparing all lanes:

```text
simd_vmas_f32x8 passed: 8 lanes
```

- [ ] **Step 2: Compile with the matching toolchain**

Compile the MPE source with `-mhost -O2`, the CPE source with
`-mslave -msimd -mieee -O2`, and link both objects with `-mhybrid` through
`/mnt/sda/zyx/toolchains/sw9a-sdk-overlay/bin/swgcc1307`. Treat any implicit
declaration, vector conversion, alignment, or mixed-link error as a failed
probe.

- [ ] **Step 3: Run on SW9A**

Transfer only the probe executable to `/tmp/tilelang-runs/simd-vmas-probe-20260903`
and run through `swrun -E 64 -i`. Require the exact success line and return code
zero.

- [ ] **Step 4: Record the exact compiler paths and result in README, then commit**

```bash
git add examples/sunway/probes
git commit -m "test(sunway): validate FP32x8 SIMD intrinsic"
```

If the probe fails, preserve its compiler/runtime evidence, stop before Task 6,
and keep Task 4 as the accepted G1a/G1b checkpoint.

---

### Task 6: Lower Abstract FP32x8 FMA To The Native Path

**Files:**
- Create: `tilelang/sunway/op/gemm/gemm_vmad.py`
- Modify: `tilelang/sunway/op/gemm/__init__.py`
- Modify: `tilelang/sunway/gemm_transform.py`
- Modify: `tilelang/sunway/transform.py`
- Modify: `tilelang/sunway/codegen.py`
- Modify: `testing/python/sunway/test_gemm_transform.py`
- Modify: `testing/python/sunway/test_runtime.py`

**Interfaces:**
- Consumes: verified Task 5 SIMD contract and Task 4 multi-K scalar loop.
- Produces: `lower_gemm_compute_to_simd(mod, plan)`, one S2 abstract leaf, one S3 native leaf, and generic native-helper C emission.

- [ ] **Step 1: Write failing SIMD S2 tests**

Require `mesh_2d/simd` S2 to contain one static abstract call site named
`tilelang_sunway_fma_f32x8` inside loops over tile M, K, and two N vectors.
Require no scalar C store in that compute region. G0 and G1 scalar S2 must retain
ordinary FP32 multiply/add.

- [ ] **Step 2: Implement the canonical compute rewrite**

Structurally match the lowered canonical scalar GEMM store and its enclosing
M/N/K loop nest. Rebuild only that compute region as:

```text
for mi in 0..tile_m:
  for kk in 0..tile_k:
    for nv in 0..tile_n/vector_width:
      tilelang_sunway_fma_f32x8(
          A_shared[mi, kk],
          &B_shared[kk, nv * vector_width],
          &C_local[mi, nv * vector_width])
```

Reject a non-canonical loop or buffer access with an S2 diagnostic; do not
silently emit scalar code for a requested SIMD target.

- [ ] **Step 3: Write failing S3 tests and map the semantic leaf**

S3 must contain `tilelang_sunway_native_fma_f32x8` and no
`tilelang_sunway_fma_f32x8`. Verifiers require vector width 8, FP32 buffers,
32-byte aligned offsets, and the native call under the K-panel loop.

- [ ] **Step 4: Emit the target-native helper structurally**

When the S3 module contains the native FP32x8 leaf, include `simd.h`, align local
arrays to 32 bytes, and emit one helper that uses `floatv8`,
`simd_set_floatv8`, vector loads/stores, and `simd_vmas`. `emit_call()` forwards
the scalar and two access pointers. No code path may inspect a kernel symbol or
matrix shape.

- [ ] **Step 5: Run focused tests, full suite, Ruff, compileall, and commit**

```bash
git add tilelang/sunway/op/gemm tilelang/sunway/gemm_transform.py \
  tilelang/sunway/transform.py tilelang/sunway/codegen.py \
  testing/python/sunway/test_gemm_transform.py testing/python/sunway/test_runtime.py
git commit -m "feat(sunway): lower GEMM to FP32x8 SIMD"
```

---

### Task 7: Prove G1 SIMD On SW9A And Record Timing Separately

**Files:**
- Modify: `examples/sunway/gemm_128_k64_main.c`
- Modify: `testing/python/sunway/test_runtime.py`
- Modify: `docs/get_started/sunway.md`

**Interfaces:**
- Consumes: one official frontend selectable as scalar or SIMD and the generic AOT tooling.
- Produces: scalar/SIMD numerical evidence, separate target timing observations, and reproducible documentation.

- [ ] **Step 1: Add target timing instrumentation to the numerical main**

Require S2/S3 attributes and generated C to distinguish scalar from SIMD while
the tensor ABI, frontend function, and numerical main stay identical. The main
runs one untimed warmup, measures seven kernel calls with the target monotonic
clock, sorts the seven elapsed values, and prints the median after the numerical
success line.

- [ ] **Step 2: Generate and compile both packages from the final commit**

Use these fixed directories:

```text
/tmp/tilelang-gemm-g1-scalar-generated-20260903
/tmp/tilelang-gemm-g1-scalar-package-20260903
/tmp/tilelang-gemm-g1-simd-generated-20260903
/tmp/tilelang-gemm-g1-simd-package-20260903
```

Inspect S1/S2/S3 and compile both with the matching SWGCC-1307 roots.

- [ ] **Step 3: Run scalar and SIMD correctness separately on SW9A**

Require:

```text
gemm_128_k64 passed: M=128 N=128 K=64
```

The scalar and SIMD deployments must each print this line. Both use identical
deterministic input formulas and the same host reference; the package directory
and S2/S3 compute metadata identify the selected implementation.

- [ ] **Step 4: Measure without mixing correctness claims**

Use the seven measured post-warmup kernel invocations per package under the same
launcher and report the printed median elapsed time for scalar and SIMD
separately.
Do not state a speedup if timing noise, launcher overhead, or insufficient
repetitions prevents a fair comparison.

- [ ] **Step 5: Document the exact commands and G1 boundaries**

Add a `Distributed Multi-K SIMD G1` subsection covering schedule flags,
generation, packaging, deployment, correctness outputs, and measured timing.
State that G1 still uses independent synchronous DMA and has no packed-weight or
PyTorch path.

- [ ] **Step 6: Run final verification and commit**

```bash
cmake --build build -j2
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 \
  -m pytest testing/python/sunway -q
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 \
  -m ruff check tilelang/sunway testing/python/sunway examples/sunway
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 \
  -m compileall -q tilelang/sunway testing/python/sunway examples/sunway
git diff --check
git add docs/get_started/sunway.md examples/sunway/gemm_128_k64_main.c \
  testing/python/sunway/test_runtime.py
git commit -m "docs(sunway): record distributed SIMD GEMM G1"
```

---

## Completion Gate

- [ ] G0 single-CPE scalar examples remain reproducible.
- [ ] G1 S2 proves complete, non-overlapping 2D mesh ownership.
- [ ] The `8 by 8` output grid activates all 64 CPEs.
- [ ] Two K panels accumulate before one output DMA.
- [ ] The standalone SIMD probe compiles and passes on SW9A.
- [ ] SIMD S3 and generated C use the verified FP32x8 path.
- [ ] Scalar and SIMD packages pass the same deterministic reference on SW9A.
- [ ] Timing is recorded separately from correctness.
- [ ] Dell build, full Sunway tests, Ruff, compileall, and diff checks pass.
- [ ] Final branch and fork contain only reviewed commits; remote host artifacts remain outside Git.
