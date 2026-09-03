# Sunway GEMM G1 Distributed Multi-K SIMD Design

## Status

Approved for inline implementation after the G0 target-validation gate. G1
extends the verified official TileLang frontend and AOT path; it does not add a
new user-facing DSL.

## Scope

G1 adds three schedule capabilities in order:

1. distribute output tiles across the configured CPE mesh;
2. accumulate more than one K panel without clearing partial results;
3. replace the scalar inner compute with the SWGCC-1307 FP32 vector path.

The frontend remains composed of `T.Kernel`, shared and fragment allocations,
`T.Pipelined`, `T.copy`, and `T.gemm`. G1 does not add double buffering,
row/column communication, packed weights, tails, dynamic shapes, FP16/BF16, or
PyTorch packaging.

## Ownership Alternatives

### Selected: two-dimensional mesh stride

For an `R by C` CPE mesh, CPE `(pe_row, pe_col)` owns output tiles:

```text
by = pe_row + q * R
bx = pe_col + p * C
```

This is complete and non-overlapping for every rectangular output-tile grid.
An `8 by 8` tile grid activates all 64 CPEs exactly once. Larger grids give each
CPE multiple tiles. Smaller grids leave only CPE coordinates that intersect the
grid active.

This mapping is selected because future row communication can share an A panel
among CPEs with equal `pe_row`, while column communication can share a B panel
among CPEs with equal `pe_col`.

### Rejected for G1: flattened cyclic ownership

`tile_id = pe_id + q * 64` uses more CPEs for very skinny grids, but destroys the
direct row/column relationship needed by the planned cooperative schedule. It
may later remain as a tuning alternative for skinny matrices.

### Rejected for G1: modulo guard over all tiles

Guarding serial tile loops with `tile_id % 64 == pe_id` is simple, but every CPE
examines every output tile. It proves ownership but does not materialize the
schedule that should be measured.

## Target Schedule Configuration

`SunwayTargetConfig` gains backend schedule fields rather than G-stage names:

```text
gemm_ownership = "single" | "mesh_2d"
gemm_compute   = "scalar" | "simd"
```

Defaults remain `single` and `scalar`, preserving G0 as a reproducible oracle.
The G1 target uses `mesh_2d` and first `scalar`, then `simd`. Unsupported values
are rejected during target normalization.

## Progressive IR Contract

### S1

S1 is unchanged. It retains the official TileLang tile grid, logical worker
domain, multi-K `T.Pipelined` loop, TileOps, and memory scopes.

### S2

The G1 S2 schedule contains:

- one semantic PE-id binding;
- derived `pe_row` and `pe_col` expressions;
- serial `bx` and `by` loops whose starts and strides implement mesh ownership;
- one clear before the K-panel loop;
- one or more K panels with A/B DMA issue and wait;
- either scalar FP32 multiply/add or an abstract FP32x8 FMA leaf;
- one C DMA put after all K panels complete.

The plan records global M/N/K, output tile counts, K-panel count, mesh shape,
active CPE count, ownership kind, compute kind, vector width, and LDM bytes.
`active_cpes` is:

```text
min(block_tiles_m, cpe_rows) * min(block_tiles_n, cpe_cols)
```

The S2 verifier proves ownership from loop starts, extents, and steps. It also
checks complete K coverage and requires vector width to divide `tile_n` for the
SIMD schedule.

### S3

The existing semantic DMA mapping remains shared. G1 additionally resolves the
abstract FP32x8 FMA leaf to a validated native vector helper. S3 retains no
TileOps and no abstract SIMD leaf.

The helper is generated structurally and uses the installed SWGCC-1307 contract:

```c
floatv8 simd_vmas(floatv8 a, floatv8 b, floatv8 c);
```

It broadcasts one A scalar, loads eight contiguous B and C values, invokes
`simd_vmas`, and stores eight C values. B and C vector addresses must be aligned
to 32 bytes. Generated LDM arrays carrying vector accesses are also aligned to
32 bytes.

The common C emitter remains independent of GEMM names and matrix shapes. It
only knows how to emit a target-native vector leaf and its required helper.

## Multi-K Ordering

For every owned output tile, C is cleared exactly once. For each K panel:

1. DMA the A and B panel to private LDM;
2. wait for both panel transfers;
3. accumulate the complete panel into C;
4. continue to the next panel without clearing C.

C is written to global memory only after the final panel. G1 uses synchronous
DMA, so one A buffer, one B buffer, and one C accumulator remain sufficient.

## Toolchain Probe

Header inspection is not target proof. Before enabling the native mapping, a
small repository probe must be compiled with the same `-mslave -msimd`
SWGCC-1307 path and run on SW9A. It verifies vector load, scalar broadcast,
`simd_vmas`, vector store, and numerical output. If this probe fails, G1 stops
after distributed multi-K scalar correctness; it must not substitute a guessed
intrinsic spelling.

## Validation Shapes

G1 keeps the two G0 shapes as regressions and adds:

```text
M = 128, N = 128, K = 64
BM = 16, BN = 16, BK = 32
output tile grid = 8 by 8
K panels = 2
```

This one shape proves all 64 CPE coordinates, multi-K accumulation, and two
FP32x8 vectors per output row. Scalar and SIMD packages use identical
deterministic inputs and independently compare against the same host reference.

## Source Organization

```text
tilelang/sunway/target.py
tilelang/sunway/op/gemm/plan.py
tilelang/sunway/op/gemm/gemm_vmad.py
tilelang/sunway/gemm_transform.py
tilelang/sunway/codegen.py
testing/python/sunway/gemm_cases.py
testing/python/sunway/test_gemm_transform.py
testing/python/sunway/test_runtime.py
examples/sunway/probes/simd_vmas_f32x8_*.c
examples/sunway/gemm_128_k64.py
examples/sunway/gemm_128_k64_main.c
```

Operator-specific scheduling stays in the GEMM package. Semantic-to-native ABI
rewriting and structural C emission remain shared backend services.

## Acceptance Gate

G1 is complete only when:

- the existing G0 frontend examples and all Sunway regressions still pass;
- S2 proves two-dimensional complete, non-overlapping mesh ownership;
- an `8 by 8` tile grid reports and exercises 64 active CPEs;
- `K=64`, `BK=32` produces two ordered accumulation panels;
- the repository SIMD probe compiles and runs on SW9A;
- S3 contains the validated FP32x8 native path and generated C calls
  `simd_vmas` without kernel-name or shape branches;
- scalar and SIMD `128x128x64` packages compile with SWGCC-1307 and pass the
  same deterministic numerical reference on SW9A;
- correctness and timing are reported separately, with no speedup claim unless
  both variants are measured under the same launch conditions.
