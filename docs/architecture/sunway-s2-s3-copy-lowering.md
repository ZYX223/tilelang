# Sunway S2/S3 Copy Lowering

## Scope

This change strengthens the existing Paper-1 `T.copy` path. It does not add a
second compiler IR, GEMM scheduling, double buffering, SIMD lowering, or a new
runtime protocol.

## Stage Contract

- S1 is TileLang TIR with `T.copy` intent still visible.
- S2 is Sunway semantic TIR. CPE ownership, LDM storage, transfer sizes, and
  abstract DMA/reply operations are explicit, but no `athread_*` ABI call is
  present.
- S3 is codegen-ready loop-buffer TIR. Ownership is unambiguous, semantic calls
  have been lowered to the SW9A ABI, and phase legality has been verified.
- C codegen is mechanical from S3 and must not make scheduling decisions.

TIR remains the authoritative representation. A typed copy plan is an analysis
result used to construct and verify TIR, not a parallel IR.

## S2 Planning

For a static contiguous copy, analysis derives:

- tensor element count and element size;
- DMA alignment in elements;
- an aligned LDM tile bounded by the per-CPE LDM budget;
- the number of logical tiles and active CPEs;
- a grid-stride ownership rule in which logical tile `i` belongs to
  `i % cpe_count`;
- the final tile transfer length.

The current MVP keeps a conservative restriction that the total transfer size
must satisfy DMA alignment. It supports a shorter final tile when that tile is
still aligned. Unaligned scalar fallback is a later feature.

S2 emits a fixed-size LDM tile buffer and a per-CPE loop over owned logical
tiles. DMA get, wait, put, and wait remain abstract semantic calls.

## S3 Lowering And Verification

S3 lowers only already-planned semantic leaves:

- PE identity to `_MYID`;
- abstract DMA get/put to `athread_get`/`athread_put`;
- abstract reply wait to the backend reply-wait operation.

Verification rejects phase violations, remaining TileLang copy operations,
remaining semantic calls after S3, native ABI calls in S2, invalid DMA
alignment, CPE ownership gaps, and LDM over-budget plans. Diagnostics identify
the failed phase and invariant.

## Acceptance

The existing 128-element copy remains unchanged at the API level. Tests also
cover a transfer requiring multiple logical tiles per CPE, an aligned short
final tile, DMA misalignment rejection, LDM-budget rejection, S2/S3 call
boundaries, and structural C emission for the tile loop.
