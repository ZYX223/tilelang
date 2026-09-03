# Sunway Backend Structure

The Sunway backend follows TileLang's existing separation between backend
registration, target-independent compiler orchestration, operator lowering,
code generation, and runtime integration. The Paper-1 implementation remains
an AOT backend: TileLang and TVM run on the development host, while generated
MPE/CPE sources are compiled with the Sunway toolchain and executed on SW9A.

## Compiler flow

```text
TileLang PrimFunc
  -> backend.py / target.py
  -> pipeline.py
  -> S1 annotated TIR
  -> op/dispatch.py
  -> op/<name>/ lowering
  -> S2 Sunway semantic TIR
  -> transform/ semantic lowering
  -> S3 native TIR
  -> codegen/
  -> MPE C + CPE C + common header + manifest
  -> SWGCC-1307
  -> SW9A runtime
```

S1, S2, and S3 are progressive TVM TIR modules. Dump files and Python plan
objects are diagnostics; they are not parallel compiler IRs.

## Ownership boundaries

| Path | Responsibility |
| --- | --- |
| `backend.py` | Register target support, pass pipeline, codegen, and execution backend. |
| `target.py` | Parse and validate Sunway target configuration. |
| `pipeline.py` | Sequence S1, S2, and S3 without owning operator rules. |
| `tir_utils.py` | Small TIR traversal helpers shared across layers. |
| `transform/` | Generic annotation and Sunway semantic-to-native conversion. |
| `op/dispatch.py` | Select the operator-owned S1-to-S2 lowering path. |
| `op/<name>/plan.py` | Derive tile, CPE ownership, LDM, and schedule decisions. |
| `op/<name>/lower.py` | Rewrite TileLang operations into Sunway semantic TIR. |
| `op/<name>/schedule.py` | Build non-trivial operator schedules when needed. |
| `op/<name>/verify.py` | Enforce operator-specific S2/S3 legality. |
| `codegen/model.py` | Extract kernel and ABI metadata from lowered TIR. |
| `codegen/cpe.py` | Emit CPE C from native TIR. |
| `codegen/mpe.py` | Emit the common header and MPE launch wrapper. |
| `codegen/project.py` | Write source files and the AOT manifest. |
| `runtime/` | Compile, package, deploy, load, and invoke generated libraries. |

## Adding an operator

Add `op/<name>/plan.py`, `lower.py`, and `verify.py`; add `schedule.py` only
when schedule construction is substantial. Register operation detection and
dispatch in `op/dispatch.py`. Extend generic semantic lowering only when the
operator introduces a reusable Sunway intrinsic, and extend codegen only when
native TIR contains a genuinely new construct.

This keeps operator growth out of `pipeline.py` and avoids one handwritten C
template per frontend kernel. CPE code emission stays structural: scheduling
decisions are represented in S2/S3 TIR before the emitter sees them.
