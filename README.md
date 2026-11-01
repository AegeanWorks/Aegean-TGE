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
.\build\release\bin\tge.exe gguf <model.gguf>
.\build\release\bin\tge.exe generate --model <model.gguf> --tokens "785 6722 315 9625 374" --n 129
.\build\release\bin\tge.exe bench gemv --rows 4096 --cols 4096 --repeat 5
.\build\release\bin\tge.exe bench gemm --m 1024 --n 1024 --k 1024 --repeat 3
.\build\release\bin\tge.exe bench mem --size-mb 256 --repeat 5
```

`tge info` prints the detected CPU, its instruction sets, the P/E core split, the
cache hierarchy, system memory, and the resulting memory budget.

`tge generate` runs real greedy decoding on a GGUF model and reports prefill and
decode token rates. Token ids are supplied directly, so a tokenizer is not
needed to measure speed; `tge gguf` dumps the metadata and tensors.

`tge bench gemv` measures the decode shape in isolation: one activation vector
through a weight matrix, streamed once. `tge bench gemm` measures prompt
processing. `tge bench mem` measures the raw bandwidth ceiling.

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
│   └── tge/
│       ├── core/             <- detection, dispatch, thread pool
│       ├── cpu/              <- kernel signatures and dispatch types
│       ├── memory/           <- RAM query, allocator, budget
│       ├── models/           <- GGUF reader and Qwen2 decoder
│       └── tensor/           <- reference operation declarations
├── src/
│   ├── core/                 <- cpu detection, dispatch, thread pool
│   ├── cpu/
│   │   ├── avx2/             <- ISA-specific, compiled with -mavx2 -mfma
│   │   └── scalar/           <- portable fallbacks for every ISA tier
│   ├── memory/               <- system RAM, aligned allocation, budget
│   ├── models/               <- GGUF reader and the Qwen2 decoder
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
| RAM | 16 GiB, single DIMM, 25-29 GB/s threaded read |
| GPU | AMD Radeon RX 9060 XT, 15.67 GiB VRAM (Vulkan only; no HIP SDK) |
| **Decode, Qwen2.5-0.5B Q8_0** | **30-33 t/s (llama.cpp: 15.2 t/s, same file, same threads)** |
| GEMV f32 4096² | 18.4 GB/s on 1 thread, 35.4 GB/s on 16 |
| f32 GEMM 1024³ | 66.8 GFLOPS single-thread, 206.6 GFLOPS on 16 threads |
| Tests | 35 cases, all passing |

Decode output was validated token for token against llama.cpp on the same model.
Prompt processing is **not** comparable yet: TGE feeds prompt tokens one at a
time (~11 t/s) while llama.cpp batches them (~279 t/s). Details and caveats:
[docs/benchmarks.md](docs/benchmarks.md).

## 🛠️ AIKit Commands (run from the AIKit folder)

| Role | Command |
|---|---|
| Status, task and gates | `python aikit.py status aegean-tge` |
| Phase 4: every gate | `python aikit.py check aegean-tge` |
| Repair memory and map, then check | `python aikit.py fix aegean-tge` |
| Resume an interrupted task | `python aikit.py resume aegean-tge` |

## 📜 License

Free for personal use under the Aegean License 1.1 (AL-1.1); see [LICENSE](LICENSE). Any commercial use needs the Aegean Commercial License 1.0 (ACL-1.0); see [COMMERCIAL.md](COMMERCIAL.md).
