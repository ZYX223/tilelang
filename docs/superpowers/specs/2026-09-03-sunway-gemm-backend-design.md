# Sunway GEMM Backend Design

## Status

Approved direction for the Paper-1 Sunway backend. This specification defines
the next compiler vertical slice after the verified multi-tile copy path.

## Goal

Lower the existing TileLang `T.gemm` operation to correct and tunable SW9A
MPE/CPE code without embedding a GEMM algorithm in the C source emitter.

The completed slice must demonstrate, in order:

1. a scalar correctness fallback on a real SW9A target;
2. multi-K tiled accumulation;
3. a compiler-selected SIMD microkernel;
4. legal two-stage A/B LDM buffering;
5. offline schedule measurement and selection.

## Scope

Paper-1 GEMM supports one CG with a configurable rectangular CPE mesh, static
M/N/K, two-dimensional dense buffers, `transpose_A` and `transpose_B`, and
FP32 correctness first. Additional storage and accumulator dtypes are enabled
only after the matching SWGCC intrinsic and numerical behavior are verified.

The first optimized path assumes row-major dense input or an explicitly packed
weight layout. Arbitrary strides, dynamic shapes, split-K across CGs, sparse
GEMM, quantized GEMM, and automatic model-wide fusion are not in this slice.

For LLM inference, prefill GEMM and small-M decode GEMV are separate schedule
families behind the same logical operation. GEMV is not silently represented by
an unsuitable large-M GEMM schedule.

## User Contract

The primary frontend is the same tiled TileLang program shape used by existing
backends. Sunway does not introduce a separate `SW.gemm` operation or redefine
`T.gemm` as a whole-matrix library call:

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, num_workers, num_stages):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float32"),
        B: T.Tensor((K, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=num_workers,
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float32")
            B_shared = T.alloc_shared((block_K, block_N), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm
```

The language-level compatibility contract covers `T.Kernel`, `T.alloc_shared`,
`T.alloc_fragment`, `T.clear`, `T.Pipelined`, `T.copy`, `T.gemm`, and the normal
buffer-region syntax. Existing target-independent parameters and annotations
retain their meanings. Sunway-specific schedule choices remain backend config
or tuner output rather than new frontend operations.

Interface compatibility does not mean that one schedule configuration is legal
on every target. CUDA thread counts, warp policies, shared-memory capacities,
and pipeline depths are target constraints. For the first single-CG Sunway
implementation, `num_workers` must equal `cpe_rows * cpe_cols`, and
`num_stages` must be one or two. Cross-backend examples should parameterize
these values while retaining the same TileLang source structure.

The Sunway meanings of the official scopes are:

- `T.Kernel` describes the logical output-tile grid and worker domain. The tile
  grid becomes CPE-owned outer loops, and the logical worker id becomes the CPE
  id; it does not create CUDA blocks or threads in generated code.
- `T.alloc_shared` describes a cooperatively visible tile. Because CPE LDM is
  physically private, layout inference realizes the logical tile through legal
  per-CPE distribution, replication, DMA, and later row/column communication.
- `T.alloc_fragment` describes each worker's accumulator fragment and lowers to
  registers or private LDM according to the selected microkernel.
- `T.Pipelined` expresses overlap intent. The Sunway schedule chooses concrete
  LDM stage buffers and DMA reply dependencies.

The initial GEMM verifier supports the cooperative access patterns produced by
the canonical tiled program above. It rejects arbitrary cross-worker shared
access when the backend cannot preserve its semantics. Users still do not write
CPE ids, DMA reply counters, SWGCC SIMD builtin names, or ping-pong pointers.

A future whole-matrix convenience template may generate this official tiled
program, but the bare `T.gemm(A_global, B_global, C_global)` shorthand is not a
Paper-1 frontend contract.

## Cross-Backend Compatibility Requirements

Sunway compilation must preserve the existing language surface and reject
unsupported schedules explicitly. In particular:

- no public Sunway-only GEMM, copy, pipeline, allocation, or synchronization
  function is required for the MVP;
- the generic `GemmWarpPolicy` argument remains accepted as a partition hint;
  its square/full-row/full-column choices map to logical CPE mesh ownership;
- common TileLang analysis and `LowerTileOp` dispatch remain in the path;
- target-specific behavior begins at implementation selection, layout
  realization, schedule legality, and semantic lowering;
- a program that compiles for multiple targets has the same mathematical and
  memory-dependence semantics, even though its physical storage, worker mapping,
  and emitted instructions differ.

## Integration With TileLang GEMM Dispatch

Sunway follows the existing CPU, CUDA, ROCm, and Metal organization:

1. The generic frontend creates `tl.tileop.gemm`.
2. C++ target dispatch selects a Sunway instruction key.
3. The Python GEMM registry resolves the instruction key to a Sunway
   `GemmBase` implementation.
4. The implementation returns TIR statements, not C source text.
5. The Sunway pipeline progressively lowers and verifies those statements.

The initial instruction keys are:

- `sunway.scalar`: scalar correctness fallback for all supported FP32 shapes;
- `sunway.vmad`: tiled SIMD path selected only when dtype, alignment, tile, and
  target-intrinsic constraints are legal.

No `gemm_compiler.py` or kernel-name-based C template is introduced. A
dedicated GEMM implementation file is expected because it owns target-specific
layout and instruction selection, but the common C emitter remains independent
of the operator kind.

## Progressive IR Contract

### S1: Annotated TileLang TIR

S1 retains `tl.tileop.gemm` and its semantic information:

- A, B, and C regions;
- M, N, and K;
- input, accumulator, and output dtypes;
- transpose flags and `clear_accum`;
- the `T.Kernel` tile grid and logical worker domain;
- shared and fragment scopes, pipeline loops, and copy dependencies.

No LDM layout, DMA ABI call, SIMD builtin, or fixed schedule is present.

### S2: Sunway Semantic TIR

S2 contains the selected schedule and enough structure to prove legality:

- output-tile ownership for every logical CPE;
- `MR`, `NR`, and `BK` micro-tile dimensions;
- outer M/N tile loops and a multi-K loop;
- typed LDM buffers for A, B, and the C accumulator;
- abstract DMA issue/wait operations and explicit reply-token dependencies;
- abstract scalar-FMA or vector-FMA computation;
- explicit pipeline stage indices when staging is enabled;
- optional abstract row/column communication, introduced only after the
  corresponding SW9A interfaces are verified.

S2 must not contain `athread_*`, `_MYID`, guessed SWGCC builtin spellings, or C
source fragments.

### S3: Codegen-Ready Loop/Buffer TIR

S3 resolves every semantic leaf to the validated SW9A ABI:

- logical worker id to the target CPE id operation;
- DMA issue and wait to the target runtime calls and reply objects;
- SIMD FMA to an intrinsic confirmed in the installed SWGCC-1307 headers;
- pipeline stages to concrete LDM buffers, reply counters, and swap logic;
- row/column communication to validated target APIs when that path is enabled;
- edge handling to guards, zero fill, or a scalar fallback.

S3 contains all scheduling decisions. Code generation is a mechanical traversal
of S3 loops, expressions, buffers, calls, and host/device regions.

## Schedule Model

`SunwayGemmPlan` is a typed analysis result, not a second compiler IR. It records:

- normalized dimensions, regions, strides, transpose flags, and dtypes;
- CPE mesh dimensions and output ownership;
- `MR`, `NR`, `BK`, vector width, unroll factor, and pipeline stages;
- per-stage A/B tile bytes, accumulator bytes, reply bytes, argument bytes, and
  total LDM use per CPE;
- DMA mode, alignment, row count, row bytes, and source stride;
- tail policy and packed-weight requirement;
- an immutable schedule id used by dumps, manifests, and tuning records.

For an 8 by 8 mesh, logical CPE `(r, c)` owns one or more output micro-tiles.
Within one CG-level output tile, each CPE accumulates an `MR by NR` C tile over
successive `BK` slices. The exact mesh dimensions come from target config rather
than hard-coded constants.

The double-buffer LDM constraint is evaluated per CPE:

```text
stages * (A_tile_bytes + B_tile_bytes)
+ C_accumulator_bytes
+ reply_and_metadata_bytes
<= configured_ldm_bytes_per_cpe
```

The configured default may be 64 KiB, but compilation and published results
must use a value validated for the selected target generation. Linux host memory
reports are not evidence of CPE LDM capacity.

## Transfer And Compute Strategies

### Scalar Baseline

Each CPE computes its owned C elements with a serial K loop. This path favors
coverage and serves as the correctness oracle for target lowering. It still uses
the common S2/S3 path and must not be a separate hand-written program.

### Multi-K SIMD Path

For each output micro-tile:

1. load the current A and B `BK` panels into LDM;
2. wait for their reply tokens;
3. accumulate into a register or LDM C tile with vector FMA along a legal axis;
4. advance to the next K panel without clearing the accumulator;
5. convert and write C after complete K coverage.

The SIMD implementation uses the actual vector types and builtins found in the
target toolchain. Names from manuals or earlier Sunway generations are treated
as candidates until a compile-and-run probe confirms them.

### Two-Stage Buffering

Stage 0 proves synchronous multi-K SIMD first. Stage 1 adds exactly two A/B
buffer versions. While compute consumes stage `k mod 2`, DMA may fill stage
`(k + 1) mod 2`. The verifier rejects a schedule when:

- a producer overwrites a stage before its consumer completes;
- compute reads a stage before its DMA reply completes;
- reply counters are reused without reset;
- the two-stage allocation exceeds LDM;
- the final partial K tile reads uninitialized elements.

Three-stage buffering is outside this slice until two-stage measurements show a
target-side benefit.

### Mesh Communication

The first optimized kernel may use independent per-CPE DMA to isolate SIMD and
pipeline correctness. A following schedule can nominate row leaders for A and
column leaders for B, then distribute panels through validated row/column
communication. This is an S2 schedule alternative, not a frontend API.

Independent-DMA and cooperative-mesh kernels remain separately measurable so
communication benefits are not inferred from a combined change.

## Weight Layout

`transpose_B=True` is common for model linear layers. The optimized inference
path must not perform a full weight transpose during every invocation. It uses
one of two explicit policies:

- direct strided access for the scalar fallback;
- an AOT packed-B layout organized by `BK` and `NR` panels for the SIMD path.

The package manifest records the required packed layout. Model loading or an
offline converter creates and caches packed weights. The PyTorch wrapper must
reject an incompatible layout instead of silently interpreting storage.

## Verifiers

The S2 verifier checks:

- supported rank, static dimensions, dtype, and transpose combination;
- complete and non-overlapping CPE ownership of C;
- complete K coverage and legal partial-tile policy;
- DMA direction, extent, stride, and alignment;
- LDM capacity and allocation overlap;
- SIMD width and alignment compatibility;
- reply-token issue/reset/wait ordering;
- pipeline producer/consumer hazards;
- packed-weight layout compatibility.

The S3 verifier additionally rejects residual TileLang GEMM calls, residual
Sunway semantic calls, unresolved worker bindings, and native calls whose
argument contract does not match the target ABI.

Diagnostics name the phase, schedule id, failed invariant, and relevant byte or
shape values. Invalid optimized schedules fall back to `sunway.scalar` only
when the requested semantic operation is supported; ABI and ownership failures
are compilation errors.

## Code And Runtime Boundaries

Expected source organization:

```text
src/sunway/op/gemm.cc
tilelang/sunway/op/gemm/__init__.py
tilelang/sunway/op/gemm/plan.py
tilelang/sunway/op/gemm/gemm_scalar.py
tilelang/sunway/op/gemm/gemm_vmad.py
```

Common semantic-to-native rewriting remains in the Sunway transform layer.
Common C expression, statement, vector, and call emission remains in
`tilelang/sunway/codegen.py`. Shared transfer planning should be extracted from
the copy implementation only when GEMM introduces a second proven use case.

The generated AOT project retains the current ownership model:

- the standalone MPE program initializes the athread runtime once;
- generated operator functions spawn and join CPE work but do not initialize
  process-global runtime state;
- the common header defines the generated kernel argument ABI;
- the manifest records shape, dtype, layout, schedule id, compiler identity,
  source hashes, and required runtime files.

A later PyTorch library exports one stable GEMM entry and registers it through
the existing Sunway operator wrapper. The wrapper validates tensors and packed
layout before forwarding addresses. PyTorch does not participate in CPE
scheduling or compilation.

## Offline Tuning

Tuning runs outside the SW9A Python runtime:

1. generate legal schedule candidates on the Dell TileLang/TVM environment;
2. reject candidates analytically using the S2 verifier;
3. generate S3 and C for the survivors;
4. compile with the matching SWGCC-1307 toolchain;
5. transfer AOT artifacts to SW9A and measure with the real launcher;
6. store correctness and latency records keyed by target, compiler, operation,
   shape bucket, dtype, transpose flags, and packed layout;
7. select the best correct schedule within a fixed measurement budget.

Initial tuning knobs are `MR`, `NR`, `BK`, mesh orientation, vector width,
unroll factor, DMA strategy, pipeline stages in `{1, 2}`, and packed-B layout.
Dynamic token counts use explicit M buckets. A learned cost model is deferred
until enough trustworthy target measurements exist.

## Verification And Acceptance

Every milestone includes structural tests, generated-source compilation, and a
real SW9A run. Passing Python tests or producing C does not count as target
acceptance.

### G0: Dispatch And Scalar Correctness

- the canonical tiled frontend parses without any Sunway-only language call;
- `T.Kernel`, shared/fragment allocations, copies, pipeline structure, and
  `T.gemm` remain visible at their documented stage boundaries;
- `T.gemm` resolves through generic GEMM dispatch to `sunway.scalar`.
- S1/S2/S3 dumps preserve the documented boundary.
- square and non-square FP32 shapes match a host reference within a declared
  absolute and relative tolerance.
- the generated executable compiles with SWGCC-1307 and passes through the real
  SW9A launcher.

### G1: Multi-K And SIMD

- a shape with `K > BK` proves multiple K panels accumulate correctly;
- generated S3 contains the validated SIMD intrinsic path;
- scalar and SIMD outputs are compared on identical inputs;
- a target timing result is reported separately from correctness.

### G2: Two-Stage Buffering

- S2 and S3 expose two stage buffers and verifiable reply-token flow;
- adversarial verifier tests reject overwrite, early-read, and LDM-overflow
  schedules;
- one-stage and two-stage target runs use the same shape and microkernel;
- double buffering is retained as a default only when measurements justify it.

### G3: Cooperative Mesh Schedule

- row/column target APIs are compile-and-run verified first;
- independent-DMA and cooperative schedules produce equivalent results;
- communication traffic and latency are compared without changing unrelated
  tuning parameters.

### G4: Offline Tuning And Packaging

- a reproducible tuning record selects among multiple legal schedules;
- the chosen schedule can be regenerated from its manifest;
- a packed-weight GEMM can be loaded by the Sunway PyTorch wrapper;
- an end-to-end PyTorch call reaches the generated CPE kernel on SW9A.

## Risks And Controls

- **Generic LowerTileOp contains SIMT-oriented assumptions.** Preserve the
  official `T.Kernel` interface, normalize its logical worker domain to the CPE
  mesh, and test every surviving binding; do not emit CUDA concepts in C.
- **Logical shared storage is not physically shared CPE memory.** Limit the MVP
  to verified cooperative GEMM access patterns and realize them through
  distribution or replication. Reject unsupported sharing rather than silently
  treating private LDM as globally visible.
- **Toolchain SIMD names may differ by Sunway generation.** Probe installed
  headers and compile minimal examples before locking the S3 intrinsic mapping.
- **Double buffering may not overlap as expected.** Keep one-stage and
  two-stage variants and decide from target evidence.
- **Repeated per-CPE DMA may dominate GEMM.** Treat mesh communication as an
  independent schedule with a scalar/SIMD correctness baseline.
- **Decode shapes may look like GEMM but behave like GEMV.** Dispatch small-M
  shapes to a separate schedule family.
- **AOT tuning can overfit fixed shapes.** Record shape buckets, compiler and
  target identity, and validate neighboring shapes before reusing a schedule.

## Implementation Sequence

1. Register the Sunway C++ GEMM target implementation and Python scalar class.
2. Normalize the official `T.Kernel` worker domain to one CPE mesh and route
   `T.gemm` through `LowerTileOp`.
3. Add scalar S2/S3 verification and pass a real FP32 GEMM.
4. Add `SunwayGemmPlan`, multi-K tile materialization, and tail policies.
5. Probe SWGCC SIMD support and add the verified VMAD microkernel.
6. Add two-stage A/B buffering and dependency verification.
7. Add the offline candidate runner and tuning-record schema.
8. Add packed-B packaging and PyTorch wrapper integration.
9. Add cooperative mesh communication only after its standalone API probe.
