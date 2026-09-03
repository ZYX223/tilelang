# Sunway S2/S3 Copy Lowering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the fixed one-DMA-per-CPE copy prototype into a verified, multi-tile S2/S3 lowering pipeline while keeping TIR authoritative.

**Architecture:** A typed analysis plan computes legal CPE ownership, DMA chunks, and LDM use. S2 materializes that plan as semantic TIR; S3 verifies and lowers semantic leaves to SW9A ABI operations; C codegen only emits the resulting loops and expressions.

**Tech Stack:** Python, TileLang, TVM TIRx, pytest, structural MPE/CPE C emission

**Spec:** `docs/architecture/sunway-s2-s3-copy-lowering.md`

## Global Constraints

- Preserve the public `T.copy` frontend and `target={"kind": "sunway"}` entrypoint.
- TIR is the only authoritative compiler IR; Python dataclasses are analysis results only.
- Keep all current runtime and SWPyTorch behavior unchanged.
- Restrict this change to static contiguous copy lowering on one 8x8 CPE mesh.
- Do not add double buffering or SIMD in this milestone.

---

### Task 1: Copy Schedule Analysis

**Files:**
- Create: `tilelang/sunway/analysis.py`
- Test: `testing/python/sunway/test_transform.py`

**Interfaces:**
- Produces: `SunwayCopyPlan` and `analyze_copy(source, destination, config, argument_count)`.
- Consumes: `SunwayTargetConfig` and TIR buffers extracted by the lowering pass.

- [x] **Step 1: Write failing planner tests**

Add tests showing that the planner preserves the 128-element schedule, caps a
tile to LDM for a large tensor, creates multiple grid-stride iterations, accepts
an aligned final tile, and rejects total byte counts that violate DMA alignment.

- [x] **Step 2: Run planner tests and verify the missing API failure**

Run: `python -m pytest testing/python/sunway/test_transform.py -q`

Expected: collection fails because `tilelang.sunway.analysis` does not exist.

- [x] **Step 3: Implement the immutable plan and legality checks**

`SunwayCopyPlan` records `total_elements`, `element_bytes`,
`alignment_elements`, `tile_elements`, `tile_count`, `active_cpes`,
`iterations_per_cpe`, `tile_bytes`, and `ldm_bytes`. `analyze_copy` checks
matching static shape/dtype, byte-addressable dtype, target alignment, launch
descriptor/reply overhead, and the per-CPE LDM limit.

- [x] **Step 4: Run planner tests**

Run: `python -m pytest testing/python/sunway/test_transform.py -q`

Expected: planner tests pass.

### Task 2: Materialize Multi-Tile S2 TIR

**Files:**
- Modify: `tilelang/sunway/transform.py`
- Modify: `testing/python/sunway/test_transform.py`

**Interfaces:**
- Consumes: `SunwayCopyPlan` from Task 1.
- Produces: `lower_tile_copy_to_semantic_tir(mod, config)` with explicit
  grid-stride tile ownership and abstract DMA calls.

- [x] **Step 1: Write failing S2 structural tests**

Assert that S2 contains a CPE-owned tile loop, dynamic final-tile byte count,
fixed LDM allocation, schedule metadata, abstract DMA calls, and no native
`athread_*` calls.

- [x] **Step 2: Run the S2 tests and verify they fail on the single-transfer body**

Run: `python -m pytest testing/python/sunway/test_transform.py -q`

- [x] **Step 3: Rewrite S2 construction around the plan**

Build `tile_index = pe_id + iteration * cpe_count`, guard it with
`tile_index < tile_count`, derive `valid_elements = min(tile_elements,
total_elements - offset)`, and use that extent and byte count for both semantic
DMA calls. Keep reply reset/wait ordering explicit in TIR.

- [x] **Step 4: Run the S2 tests**

Run: `python -m pytest testing/python/sunway/test_transform.py -q`

Expected: all S2 tests pass.

### Task 3: Verify And Lower S2 To S3

**Files:**
- Modify: `tilelang/sunway/transform.py`
- Modify: `tilelang/sunway/pipeline.py`
- Modify: `testing/python/sunway/test_transform.py`

**Interfaces:**
- Produces: `verify_semantic_tir(mod, config)`,
  `lower_semantic_to_native_tir(mod, config)`, and
  `verify_native_tir(mod, config)`.

- [x] **Step 1: Write failing stage-verifier tests**

Cover native calls illegally present in S2, semantic calls remaining in S3,
LDM over-budget metadata, missing ownership metadata, and a valid S2/S3 pair.

- [x] **Step 2: Run the verifier tests and observe the missing APIs**

Run: `python -m pytest testing/python/sunway/test_transform.py -q`

- [x] **Step 3: Implement phase checks and wire them into the pipeline**

Verify S2 after materialization, lower semantic leaves, mark S3, then verify S3
before codegen. Diagnostics include the phase and violated invariant.

- [x] **Step 4: Run the transform tests**

Run: `python -m pytest testing/python/sunway/test_transform.py -q`

Expected: all transform tests pass.

### Task 4: Emit S3 Tile Loops Mechanically

**Files:**
- Modify: `tilelang/sunway/codegen.py`
- Modify: `testing/python/sunway/test_backend_registration.py`

**Interfaces:**
- Consumes: S3 `For`, `Min`, arithmetic, and guarded DMA statements.
- Produces: CPE C with an ordinary grid-stride `for` loop and computed DMA
  byte count.

- [x] **Step 1: Write a failing generated-C assertion**

Lower a large copy using a small test LDM budget and assert that CPE C contains
the tile loop, tile guard, and computed byte count rather than one fixed copy.

- [x] **Step 2: Run the backend test and observe unsupported `For`/`Min` nodes**

Run: `python -m pytest testing/python/sunway/test_backend_registration.py -q`

- [x] **Step 3: Add mechanical statement and expression emission**

Teach `_CPEEmitter` to emit TIR `For` and `Min` nodes. Do not add schedule
selection or kernel-name dispatch to codegen.

- [x] **Step 4: Run backend tests**

Run: `python -m pytest testing/python/sunway/test_backend_registration.py -q`

Expected: existing and large-copy codegen tests pass.

### Task 5: Regression And Target Validation

**Files:**
- Modify only if a verified defect is found in the files above.

**Interfaces:**
- Consumes: completed S2/S3 pipeline and current AOT runtime.
- Produces: local/A100 test evidence and, when reachable, Dell/SW9A compile-run evidence.

- [x] **Step 1: Run all Sunway Python tests**

Run: `python -m pytest testing/python/sunway -q`

- [x] **Step 2: Generate the canonical copy project**

Run: `python examples/sunway/copy_128.py --output-dir /tmp/tilelang-copy-128`

Check all S1/S2/S3 dumps plus MPE C, CPE C, common header, and manifest.

- [x] **Step 3: Compile with the matching SWGCC-1307 toolchain**

Use the existing Dell toolchain adapter and record compiler output without
changing runtime packaging.

- [x] **Step 4: Run on SW9A**

Deploy the compiled package to the compute node, run through `swrun`, and require
the existing `copy_128 passed: 128 elements` result. If the target is unreachable,
report this separately from compiler and unit-test status.
