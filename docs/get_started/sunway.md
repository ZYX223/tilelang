# Sunway AOT Workflow

The Sunway backend uses an ahead-of-time workflow. TileLang and TVM run on a
development or cross-compilation host; the SW9A compute node only needs the
Sunway runtime, SWPython, SWPyTorch, and the packaged artifacts. A dedicated
login node is therefore optional when the cross-compilation host can reach the
compute node directly.

## Host roles

| Host | Responsibility | Required software |
| ---- | -------------- | ----------------- |
| Development/build host | Run the Python DSL, lower TIR, generate C, and cross-compile the bundle | TileLang/TVM, host PyTorch, matching SWGCC and copied SWPython/SWPyTorch SDK files |
| SW9A compute node | Load SWPyTorch, register the operator, and execute MPE/CPE code | SWPython, SWPyTorch, `libathread`, and compatible system runtime libraries |

The compiler SDK and target runtime must match. Building successfully with a
different SWGCC release does not prove that its CPE image can execute on the
target node.

## 1. Lower the TileLang program

```python
import tilelang
import tilelang.language as T


@T.prim_func
def copy_128(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
    T.copy(A, B)


artifact = tilelang.lower(
    copy_128,
    target={"kind": "sunway", "output_dir": "generated/copy_128"},
    runtime_only=True,
)
```

The output directory contains three inspection dumps and the AOT source bundle:

- `s1_annotated_tir.txt`: target-independent computation plus Sunway semantic annotations.
- `s2_semantic_tir.txt`: tile decomposition, DMA operations, and CPE ownership.
- `s3_lowered_tir.txt`: flattened buffers, concrete LDM use, synchronization, and low-level loops.
- `mpe_copy_128.c`, `cpe_copy_128.c`, and `copy_128_common.h`: cross-compiler inputs.
- `manifest.json`: tensor ABI and artifact metadata.

The generated MPE function launches and joins the CPE entry. It does not call
`athread_init`; the owning executable must initialize CRTS exactly once.

## 2. Build an SWPyTorch bundle

```bash
python examples/sunway/package_copy_128.py \
  --generated-dir generated/copy_128 \
  --package-dir packages/copy_128 \
  --toolchain-root /path/to/swgcc-sdk \
  --overlay-root /path/to/sw9a-sdk-overlay \
  --swtorch-sdk-root /path/to/copied-swtorch-sdk \
  --artifact torch
```

`SunwayLibraryGenerator` builds the mixed MPE/CPE kernel library, the boxed
PyTorch registration library, and `tilelang_swpython`. The launcher is linked
with the kernel library as a startup dependency. This is required because the
SWGCC dynamic runtime maps CPE text from dependencies present at process start;
loading a new mixed MPE/CPE library after Python starts is too late.

The runtime package must contain:

- `tilelang_swpython`
- the generated kernel library, such as `libcopy_128.so`
- the PyTorch registration library, such as `tilelang_sunway_copy_128_ops.so`
- `tilelang_sunway_adapter.py` and `tilelang_sunway_torch.py`
- the invocation script and `manifest.json`

## 3. Deploy directly to SW9A

```bash
python examples/sunway/deploy_copy_128.py \
  --package-dir packages/copy_128 \
  --remote-host user@sw9a-compute
```

`SunwaySSHExecutor` copies the package into an isolated remote directory,
sources the SWPython environment, sets the dynamic CPE segment sizes, and runs
`tilelang_swpython` directly. It intentionally does not put this Python path
inside `swrun`: both `swrun` and the `-mdynamic` launcher would otherwise try to
own the same stask/CRTS initialization.

Standalone generated executables remain a separate path and can still be run
with the configured `swrun` launcher. Their owning `main` must call
`athread_init()` exactly once before invoking the first generated operator; the
operator functions themselves remain reusable and do not initialize CRTS.

## Scalar GEMM G0

G0 is the first end-to-end correctness baseline for the official TileLang tiled
GEMM frontend. The same frontend constructs used by other backends are retained:
`T.Kernel`, `T.alloc_shared`, `T.alloc_fragment`, `T.Pipelined`, `T.copy`, and
`T.gemm`. Sunway-specific ownership, DMA, LDM, and native calls are introduced by
the backend after the frontend program has been parsed.

Generate the square and non-square projects on the Dell build host:

```bash
cd /mnt/sda/zyx/project/tilelang-paper1-clean

/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 \
  examples/sunway/gemm_32.py \
  --output-dir /tmp/tilelang-gemm-32-generated-20260903

/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 \
  examples/sunway/gemm_m32_n16_k32.py \
  --output-dir /tmp/tilelang-gemm-m32-n16-k32-generated-20260903
```

Each directory contains the S1, S2, and S3 TIR dumps, MPE C, CPE C, a common
header, a manifest, and the numerical test `main`. S1 retains the tiled TileLang
program, S2 exposes semantic DMA and scalar arithmetic, and S3 contains the
native Sunway leaves consumed by C code generation.

Cross-compile both projects with the SWGCC-1307 toolchain and matching SDK
overlay:

```bash
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 \
  examples/sunway/package_aot.py \
  --generated-dir /tmp/tilelang-gemm-32-generated-20260903 \
  --package-dir /tmp/tilelang-gemm-32-package-20260903 \
  --toolchain-root /mnt/sda/zyx/toolchains/swgcc710-tools-SEA-1307 \
  --overlay-root /mnt/sda/zyx/toolchains/sw9a-sdk-overlay

/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 \
  examples/sunway/package_aot.py \
  --generated-dir /tmp/tilelang-gemm-m32-n16-k32-generated-20260903 \
  --package-dir /tmp/tilelang-gemm-m32-n16-k32-package-20260903 \
  --toolchain-root /mnt/sda/zyx/toolchains/swgcc710-tools-SEA-1307 \
  --overlay-root /mnt/sda/zyx/toolchains/sw9a-sdk-overlay
```

Deploy the packages directly and run them through `swrun -E 64 -i` on SW9A:

```bash
/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 \
  examples/sunway/run_aot.py \
  --package-dir /tmp/tilelang-gemm-32-package-20260903 \
  --remote-host root@10.10.10.22 \
  --remote-root /tmp/tilelang-runs \
  --executable gemm_32 \
  --deployment-id gemm-g0-scalar-20260903

/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3 \
  examples/sunway/run_aot.py \
  --package-dir /tmp/tilelang-gemm-m32-n16-k32-package-20260903 \
  --remote-host root@10.10.10.22 \
  --remote-root /tmp/tilelang-runs \
  --executable gemm_m32_n16_k32 \
  --deployment-id gemm-g0-nonsquare-20260903
```

Successful target output is:

```text
gemm_32 passed: M=32 N=32 K=32
gemm_m32_n16_k32 passed: M=32 N=16 K=32
```

G0 deliberately assigns all output tiles to CPE 0 and uses scalar FP32
multiply-add. It proves frontend compatibility, progressive lowering, generated
MPE/CPE compilation, deployment, and numerical correctness. It is not a
performance result. G1 below distributes ownership across 64 CPEs and adds
multi-K tiling and SIMD; later stages add double buffering, mesh communication,
and offline tuning.

## Distributed Multi-K SIMD GEMM G1

G1 keeps the same official TileLang frontend and selects the Sunway schedule
through target configuration. The `gemm_128_k64` example uses an `8 x 8` output
tile grid, `16 x 16 x 32` per-CPE tiles, and two K panels:

```python
target = {
    "kind": "sunway",
    "gemm_ownership": "mesh_2d",
    "gemm_compute": "simd",
    "output_dir": output_dir,
    "output_indices": [2],
}
```

`mesh_2d` maps one output tile to each of the 64 CPE coordinates. The scalar
variant is an oracle for the same distributed and multi-K schedule. The SIMD
variant rewrites only the inner compute region to the abstract S2
`tilelang_sunway_fma_f32x8` leaf. S3 resolves it to
`tilelang_sunway_native_fma_f32x8`, and structural C emission produces the
verified `floatv8` and `simd_vmas` sequence.

Generate both variants on the Dell build host:

```bash
cd /mnt/sda/zyx/project/tilelang-paper1-clean
PY=/mnt/sda/zyx/envs/tilelang-sunway-paper1-clean/bin/python3

$PY examples/sunway/gemm_128_k64.py \
  --compute scalar \
  --output-dir /tmp/tilelang-gemm-g1-scalar-generated-20260903
$PY examples/sunway/gemm_128_k64.py \
  --compute simd \
  --output-dir /tmp/tilelang-gemm-g1-simd-generated-20260903
```

Inspect the three dumps before packaging. Both S2 files must report
`sunway.ownership = "mesh_2d"` and `sunway.k_panels = 2`; their
`sunway.compute` values distinguish scalar and SIMD. Only the SIMD S2/S3 pair
contains the abstract/native FP32x8 leaves.

Cross-compile both generated projects:

```bash
TOOLCHAIN=/mnt/sda/zyx/toolchains/swgcc710-tools-SEA-1307
OVERLAY=/mnt/sda/zyx/toolchains/sw9a-sdk-overlay

$PY examples/sunway/package_aot.py \
  --generated-dir /tmp/tilelang-gemm-g1-scalar-generated-20260903 \
  --package-dir /tmp/tilelang-gemm-g1-scalar-package-20260903 \
  --toolchain-root $TOOLCHAIN --overlay-root $OVERLAY
$PY examples/sunway/package_aot.py \
  --generated-dir /tmp/tilelang-gemm-g1-simd-generated-20260903 \
  --package-dir /tmp/tilelang-gemm-g1-simd-package-20260903 \
  --toolchain-root $TOOLCHAIN --overlay-root $OVERLAY
```

Deploy the packages separately so their correctness and timing remain
attributable:

```bash
$PY examples/sunway/run_aot.py \
  --package-dir /tmp/tilelang-gemm-g1-scalar-package-20260903 \
  --remote-host root@10.10.10.22 --remote-root /tmp/tilelang-runs \
  --executable gemm_128_k64 --deployment-id gemm-g1-scalar-repeat-20260904
$PY examples/sunway/run_aot.py \
  --package-dir /tmp/tilelang-gemm-g1-simd-package-20260903 \
  --remote-host root@10.10.10.22 --remote-root /tmp/tilelang-runs \
  --executable gemm_128_k64 --deployment-id gemm-g1-simd-repeat-20260904
```

Both packages passed the same deterministic host reference on SW9A on
2026-09-04. After one warmup, the standalone program measured seven in-process
kernel calls and printed these median observations:

```text
# scalar
gemm_128_k64 passed: M=128 N=128 K=64
gemm_128_k64 median_ms: 0.170961 over 7 runs

# SIMD
gemm_128_k64 passed: M=128 N=128 K=64
gemm_128_k64 median_ms: 0.109881 over 7 runs
```

These numbers include the MPE `athread_spawn` and `athread_join` cost but not
SSH or `swrun` process startup. They are short-run observations, not a formal
speedup claim. One transient `swrun` exit code 255 occurred before a successful
rerun of the same SIMD binary, so longer repeated measurements are required for
a performance result.

G1 still uses private per-CPE buffers and synchronous DMA. It does not implement
double buffering, row/column mesh communication, packed weights, tails, dynamic
shapes, or mixed precision. This standalone G1 also does not package a PyTorch
operator; the existing SWPyTorch AOT wrapper remains a separate integration
path.
