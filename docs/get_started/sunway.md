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
