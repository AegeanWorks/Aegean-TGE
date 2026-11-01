# Hardware Support

`tge info` prints this report for any machine. The values below were measured on
the development system with `tge info` on 2026-11-01.

## Development machine (measured)

| Component | Value | Source |
|---|---|---|
| CPU | Intel Core i5-14400F, family 6 model 191 stepping 2 | CPUID leaf 0/1 |
| Cores | 10 physical / 16 logical (6 performance, 4 efficient) | `GetLogicalProcessorInformationEx` |
| Caches | L1D 416 KiB, L2 2048 KiB, L3 20480 KiB | `RelationCache` |
| NUMA | 1 node | `RelationNumaNode` |
| Instructions | sse2, sse3, ssse3, sse4.1, sse4.2, avx, avx2, fma, **avx-vnni** | CPUID + XCR0 |
| RAM | 16.00 GiB installed, 15.86 GiB visible | `GlobalMemoryStatusEx` |
| GPU | AMD Radeon RX 9060 XT, driver 26.3.1 (LLPC) | `vulkaninfo` |
| VRAM | 15.67 GiB device-local heap + 7.93 GiB host-visible heap | Vulkan memory heaps |
| OS | Windows 11 Pro 10.0.26200 | `Win32_OperatingSystem` |

### Notes that change the plan

- **AVX-512 is not available.** The report is correct: consumer Raptor Lake fuses
  AVX-512 off. AVX2/FMA and AVX-VNNI are the ceiling on this machine.
- **AVX-VNNI is available.** It is currently detected but not used. It is the
  natural next kernel after the AVX2 one, because Raptor Lake benefits from
  `vpdpbusd` for quantized dot products.
- **The specification named an i5-10400F; the machine has an i5-14400F.** This is
  better (10 cores vs 6), but it introduces hybrid P/E cores, so a naive "spawn
  one worker per logical processor" pool is already wrong on the target machine.
- **RAM is the bottleneck, not VRAM.** 16 GiB in a single DIMM against 15.67 GiB
  of VRAM. A 7B Q4 model needs roughly 4.4 GiB of weights plus KV cache; with
  1.95 GiB free during the audit it does not fit without closing applications.
  Memory-constrained mode (Mode D) is therefore the first *useful* mode, not a
  corner case.
- **A single DIMM means single-channel bandwidth.** Memory-bound decode will be
  roughly half the speed it would be with two sticks of the same size.

## Supported instruction sets

The engine targets this ladder, dispatched at runtime:

| Tier | Status |
|---|---|
| Scalar C++ reference | implemented |
| SSE2 | detected, kernels not written |
| AVX + FMA | detected, kernels not written |
| AVX2 + FMA | implemented (GEMM) |
| AVX-VNNI | detected, kernels not written |
| AVX-512 / VNNI | not present on target hardware |
| AMX | not present on target hardware |

Nothing above AVX2 can be tested on this machine, and untested code is not
claimed as working.

## GPU backends

| Backend | Status |
|---|---|
| Vulkan Compute | loader and AMD ICD present; **no SDK** (headers, `glslc`) |
| AMD HIP | **not installed**; no `HIP_PATH`, no ROCm directory |

Neither backend exists in code. `gfx1200` support in a HIP SDK must be verified
before any HIP work is scheduled, because the RX 9060 XT is new enough that
toolkit support cannot be assumed.

## Toolchain (verified)

| Tool | Version | Location |
|---|---|---|
| GCC (MinGW-w64 UCRT, POSIX threads) | 16.2.0 | winget `BrechtSanders.WinLibs.POSIX.UCRT` |
| CMake | 4.4.4 | winget `Kitware.CMake` |
| Ninja | 1.13.2 | winget `Ninja-build.Ninja` |
| MSVC | **not installed** | Build Tools 2022 has only MSBuild; no VC tools, no Windows SDK |

Consequences:

- Builds are GCC-only until MSVC is installed. HIP requires MSVC on Windows, so
  GPU work needs an admin-installed Build Tools with the C++ workload.
- Executables are linked statically (`-static`) so they run without the MinGW
  runtime DLLs on PATH.
