# Aegean TGE

<!-- aegean:media:start -->
<p align="center"><img src="assets/social-preview.svg" alt="Aegean-Tge social preview" width="100%"></p>
<p align="center"><a href="LICENSE"><img src="https://img.shields.io/badge/license-AL--1.1-2f9e44" alt="License"></a> <a href="https://github.com/AegeanWorks/AIKit"><img src="https://img.shields.io/badge/AIKit-integrated-6f42c1" alt="AIKit Integrated"></a></p>
<!-- aegean:media:end -->


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
.\build\release\bin\tge.exe generate --model <model.gguf> --prompt "The capital of France is" --n 32 --text
.\build\release\bin\tge.exe bench gemv --rows 4096 --cols 4096 --repeat 5
.\build\release\bin\tge.exe bench gemm --m 1024 --n 1024 --k 1024 --repeat 3
.\build\release\bin\tge.exe bench mem --size-mb 256 --repeat 5
```

**Before touching anything, take a snapshot** (project guideline G-001):

```powershell
.\scripts\measure.ps1 > before.txt
# ... make the change ...
.\scripts\measure.ps1 > after.txt
Compare-Object (Get-Content before.txt) (Get-Content after.txt)
```

`measure.ps1` prints the selected kernels and one line per benchmark, so the
diff shows exactly what moved. It is a single snapshot; this machine varies by
up to 20 percent run to run, so take two or three before believing a small
delta. Every change and its before/after numbers are recorded in
[docs/measurement-log.md](docs/measurement-log.md) (guideline G-002).

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
├── assets/                   <- generated logo and social preview
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
| **Decode, Qwen2.5-0.5B Q8_0** | **40-47 t/s short context; 23 t/s at 2048 (llama.cpp: 15.2 t/s, same file, same threads)** |
| **Prompt, 510 tokens** | **152-199 t/s (llama.cpp: 285.3 t/s)** |
| **Prompt, 2048 tokens** | **144-204 t/s after parallel attention** |
| GEMV f32 4096² | 18.4 GB/s on 1 thread, 35.4 GB/s on 16 |
| f32 GEMM 1024³ | 66.8 GFLOPS single-thread, 206.6 GFLOPS on 16 threads |
| Quantized formats | Q8_0, Q5_0, Q4_0, Q4_K, Q5_K, Q6_K |
| Context | 4096 by default, `--context` up to 32768 |
| Tests | 54 cases, all passing |

Decode and prompt output were both validated token for token against llama.cpp,
on Q8_0 and on Q4_K_M (which mixes four quantized formats in one file). Prompts
can be given as text (`--prompt`) or token ids (`--tokens`), and output prints as
text with `--text`; the tokenizer was validated against `llama-tokenize` on
punctuation, decimals, contractions and whitespace. Sampling supports
temperature, top-k, top-p, min-p and repetition penalty; the benchmarks use
greedy so they are reproducible. Control tokens such as `<|im_start|>` are
matched from prompt text, omitted from text output, and generation stops at the
model's end-of-sequence/end-of-turn id instead of continuing past it. Decode
hands work to a spin-then-sleep pool, and attention spreads over the pool once
the context is long enough to pay for the handoff. Numbers: [docs/measurement-log.md](docs/measurement-log.md);
method and caveats: [docs/benchmarks.md](docs/benchmarks.md).

## 🛠️ AIKit Commands (run from the AIKit folder)

| Role | Command |
|---|---|
| Status, task and gates | `python aikit.py status aegean-tge` |
| Phase 4: every gate | `python aikit.py check aegean-tge` |
| Repair memory and map, then check | `python aikit.py fix aegean-tge` |
| Resume an interrupted task | `python aikit.py resume aegean-tge` |

<!-- aegean:brand-assets:start -->
## 🎨 Brand Assets

> **AI-Generated Brand Artwork.** The banner and logo are **AI-generated brand assets**. They are illustrative and may not literally depict every feature or the full project scope.
<!-- aegean:brand-assets:end -->

## 📜 License

Free for personal use under the Aegean License 1.1 (AL-1.1); see [LICENSE](LICENSE). Any commercial use needs the Aegean Commercial License 1.0 (ACL-1.0); see [COMMERCIAL.md](COMMERCIAL.md).
