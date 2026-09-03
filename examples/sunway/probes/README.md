# Sunway SIMD Probes

`simd_vmas_f32x8` isolates the SWGCC-1307 FP32x8 contract used by the Sunway
GEMM backend. CPE 0 DMA-loads two aligned vectors, broadcasts one scalar with
`simd_set_floatv8`, evaluates `simd_vmas`, and DMA-stores eight output lanes.

The MPE and CPE translation units must be compiled separately:

```bash
SWGCC=/mnt/sda/zyx/toolchains/sw9a-sdk-overlay/bin/swgcc1307
SDK=/mnt/sda/zyx/toolchains/swgcc710-tools-SEA-1307

$SWGCC -mhost -O2 -I$SDK/shared_include \
  -c simd_vmas_f32x8_mpe.c -o simd_vmas_f32x8_mpe.o
$SWGCC -mslave -msimd -mieee -O2 -I$SDK/shared_include \
  -c simd_vmas_f32x8_cpe.c -o simd_vmas_f32x8_cpe.o
$SWGCC -mhybrid simd_vmas_f32x8_mpe.o simd_vmas_f32x8_cpe.o \
  -o simd_vmas_f32x8
```

The target acceptance line is:

```text
simd_vmas_f32x8 passed: 8 lanes
```

## Verified Environment

Verified on 2026-09-04 with:

- cross-compilation host: Dell `10.10.10.24`;
- compiler: `/mnt/sda/zyx/toolchains/sw9a-sdk-overlay/bin/swgcc1307`;
- SDK: `/mnt/sda/zyx/toolchains/swgcc710-tools-SEA-1307`;
- target: SW9A `10.10.10.22`;
- target command: `swrun -E 64 -i ./simd_vmas_f32x8`;
- result: `simd_vmas_f32x8 passed: 8 lanes`, exit code 0.
