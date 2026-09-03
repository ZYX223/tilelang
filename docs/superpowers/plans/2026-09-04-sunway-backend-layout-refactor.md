# Sunway Backend Layout Refactor Plan

**Goal:** Reorganize the validated Sunway backend around the same ownership
boundaries used by TileLang's established backends without changing frontend
syntax, S1/S2/S3 semantics, generated MPE/CPE source, manifest ABI, or runtime
behavior.

## Scope

1. Keep `backend.py`, `target.py`, and `pipeline.py` as thin registration and
   orchestration modules.
2. Move generic S1/S2/S3 transforms and phase verification into
   `tilelang/sunway/transform/`.
3. Move copy- and GEMM-specific lowering and legality checks into
   `tilelang/sunway/op/<op>/`.
4. Split the Python AOT source generator into extraction, generic C emission,
   MPE/CPE emission, and project packaging modules under
   `tilelang/sunway/codegen/`.
5. Preserve the current runtime package and public backend entry points.

## Non-goals

- No new optimization, double buffering, tuning rule, or frontend operation.
- No migration of the Python C emitter to C++ in this refactor.
- No generated-source or SW9A ABI change.

## Verification

1. Capture representative copy, scalar GEMM, and SIMD GEMM S1/S2/S3 and AOT
   sources before the move.
2. Run the complete Sunway Python test suite after each structural boundary is
   moved.
3. Compare normalized pre/post generated artifacts exactly.
4. Run Ruff, compileall, and `git diff --check`.
5. Rebuild scalar and SIMD packages with SWGCC-1307 on Dell and execute both on
   SW9A through the existing direct deployment path.
