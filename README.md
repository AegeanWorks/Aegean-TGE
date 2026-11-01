# Aegean TGE

[![Built with AIKit](https://img.shields.io/badge/built%20with-AIKit-6f42c1)](https://github.com/AegeanWorks/AIKit)

Native Windows hybrid CPU and GPU LLM inference engine in C++20 that treats system RAM and GPU VRAM as one coordinated memory tier.

Aegean TGE (Token Generation Engine) is an independent inference runtime. It owns
its model loading, memory placement and kernels; it does not wrap llama.cpp,
Ollama or any other engine.

**Status: 0.1.0 — CPU foundation.** Hardware detection, reference tensor
operations, an AVX2 GEMM with runtime dispatch, and a memory budget manager are
implemented and tested. There is no inference yet.

## ⚡ 30-Second Cold-Start Quick Reference

- **Runtime Prerequisites**: Windows 11 x64. Executables are statically linked, so no runtime DLLs or redistributables are needed.
- **Build Prerequisites**: GCC 13+ or MSVC 19.30+, CMake 3.25+, Ninja. `scripts\env.ps1` adds the per-user winget toolchain to `PATH`.
- **Canonical Regression Shield Gate**:
  ```bash
  scripts\test.cmd
  ```
- **Hardware and kernel report**:
  ```bash
  .\build\release\bin\tge.exe info
  ```

```powershell
. .\scripts\env.ps1
cmake --preset release
cmake --build --preset release
.\build\release\bin\tge.exe info
.\build\release\bin\tge.exe bench --m 1024 --n 1024 --k 1024 --repeat 3
.\build\release\bin\tge.exe bench mem --size-mb 256 --repeat 3
```

`tge info` prints the detected CPU, its instruction sets, the P/E core split, the
cache hierarchy, system memory, and the resulting memory budget. `tge bench`
measures the selected GEMM kernel at one and at all threads against the
reference, and reports the maximum numerical deviation. `tge bench mem` measures
the streaming bandwidth ceiling that decode is bounded by.

## 📦 What's Inside

```text
aegean-tge/
├── .gitignore                <- build output and models excluded
├── CMakeLists.txt            <- CMake 3.25+, C++20, per-TU ISA flags
├── CMakePresets.json         <- release and debug presets (Ninja)
├── LICENSE                   <- Aegean License 1.1 (AL-1.1): free personal use
├── COMMERCIAL.md             <- Commercial licensing terms
├── LICENSES/                 <- REUSE license texts
├── REUSE.toml                <- Machine-readable SPDX metadata
├── README.md
├── .aikit/                   <- this project's memory: record, ledger, decisions
├── apps/
│   └── cli/                  <- `tge` executable: info, bench
├── include/
│   └── tge/                  <- public headers: core, cpu, memory, tensor
├── src/
│   ├── core/                 <- cpu detection, dispatch, thread pool
│   ├── cpu/
│   │   └── avx2/             <- ISA-specific, compiled with -mavx2 -mfma
│   ├── memory/               <- system RAM, aligned allocation, budget
│   └── tensor/               <- reference operations
├── tests/                    <- dependency-free harness, run by CTest
├── docs/                     <- architecture, roadmap, hardware, benchmarks, ADRs
└── scripts/                  <- env.ps1 (toolchain), test.cmd (gate)
```

## Verified on this machine

| Measurement | Value |
|---|---|
| CPU | Intel Core i5-14400F, 10C/16T (6P + 4E), AVX2 + FMA + AVX-VNNI |
| Best ISA detected | `avx-vnni` |
| RAM | 16 GiB, single DIMM, ~12.4 GB/s read bandwidth |
| GPU | AMD Radeon RX 9060 XT, 15.67 GiB VRAM (Vulkan only; no HIP SDK) |
| f32 GEMM 1024³ | 66.8 GFLOPS single-thread, 184.9 GFLOPS on 16 threads |
| f32 GEMM 2048³ | 43.3 GFLOPS single-thread, 246.5 GFLOPS on 16 threads |
| llama.cpp baseline (0.5B Q4_K_M) | 407.6 prompt t/s, 8.73 decode t/s |
| Tests | 24 cases, all passing |

The 2048³ case is 13.5x the pre-rewrite kernel. TGE does not run inference yet,
so the llama.cpp row is a bar to clear, not a comparison.

Details and caveats: [docs/benchmarks.md](docs/benchmarks.md) and
[docs/hardware-support.md](docs/hardware-support.md).

## 🛠️ AIKit Commands (run from the AIKit folder)

| Role | Command |
|---|---|
| Status, task and gates | `python aikit.py status aegean-tge` |
| Phase 4: every gate | `python aikit.py check aegean-tge` |
| Repair memory and map, then check | `python aikit.py fix aegean-tge` |
| Resume an interrupted task | `python aikit.py resume aegean-tge` |

## 📜 License

Free for personal use under the Aegean License 1.1 (AL-1.1); see [LICENSE](LICENSE). Any commercial use needs the Aegean Commercial License 1.0 (ACL-1.0); see [COMMERCIAL.md](COMMERCIAL.md).
