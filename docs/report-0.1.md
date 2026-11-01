# Engineering Report — TGE 0.1 Foundation

- Date: 2026-11-01
- Machine: Intel Core i5-14400F / AMD Radeon RX 9060 XT / Windows 11 Pro 10.0.26200
- Scope: milestone 0.1 of the Aegean TGE specification

This report covers what was audited, what was built, what was measured and what
is still missing. Where something was not done, it says so rather than implying
completion.

Revision 2 (same day) adds the llama.cpp CPU baseline and the GEMM rewrite, and
revises sections 5, 7, 8 and 9 accordingly.

Revision 3 (same day) adds the GEMV decode kernel, multi-threaded bandwidth
measurement, and the dispatch fix for the streaming kernels. It also records the
finding that matters most: decode has roughly 4x of headroom over the measured
llama.cpp baseline on this machine.

Revision 4 (same day) adds GGUF loading and a Qwen2 decoder so the engine can be
compared against llama.cpp end to end. Decode is about **2x faster**, validated
token for token. Prompt processing is not comparable yet: it is not batched.

Revision 5 (same day) hardens the GGUF reader against malformed files, adds the
first tests for it, guards the decoder's public entry points, and records a
rejected optimization: the integer-domain Q8 inner loop measured flat and was
reverted (ADR-0006).

Revision 6 (same day) closes two of the stated caveats. Prompt processing is now
batched: **11 to 170.9 t/s**, 60 percent of llama.cpp instead of 4 percent, and
token-identical across chunk boundaries. Sampling is implemented: temperature,
top-k, top-p, min-p, repetition penalty and seed. Tokenizer and quantized types
other than Q8_0 remain open.

Revision 7 (same day) closes two more. **Q8_0, Q5_0, Q4_0, Q4_K, Q5_K and Q6_K
are all executable**, validated by running a Q4_K_M model — which mixes four of
them — and reproducing llama.cpp's output token for token. **Context is
selectable** (`--context`, default 4096, up to 32768). Token ids now **decode to
text**. Two items remain: text-to-ids encoding, and the prefill arithmetic gap.

## 1. Hardware audit

| Item | Measured |
|---|---|
| CPU | Intel Core i5-14400F, family 6 model 191 stepping 2 |
| Cores | 10 physical / 16 logical, 6 performance + 4 efficient |
| L1D | 416 KiB (6 x 48 KiB P + 4 x 32 KiB E) |
| L2 | 2048 KiB max instance |
| L3 | 20480 KiB |
| NUMA | 1 node |
| Instructions | sse2, sse3, ssse3, sse4.1, sse4.2, avx, avx2, fma, avx-vnni |
| AVX-512 | **absent** (fused off on consumer Raptor Lake) |
| AMX | absent |
| RAM | 16.00 GiB installed, 15.86 GiB visible, 1.95 GiB available at audit |
| Memory configuration | one 16 GiB Micron CP16G56C46U5.C8D at 4800 MT/s, single channel |
| GPU | AMD Radeon RX 9060 XT, device ID 0x7590, driver 26.3.1 (LLPC) |
| VRAM | 15.67 GiB device-local heap, 7.93 GiB host-visible heap, 256 MiB third heap |
| Vulkan | instance 1.4.341, device API 1.4.344 |
| Disks | C: 17.7 GB free, D: 81 GB, E: 1128 GB, G: 479 GB |

### Deviations from the specification

1. The specification named an i5-10400F (6C/12T, AVX2 only). The machine has an
   i5-14400F (10C/16T) with hybrid P/E cores and AVX-VNNI.
2. The specification assumed AVX-512 would be available on a future Intel part.
   It is not present here and cannot be tested.
3. RAM was described as 16 GB expanding to 64 GB. It is 16 GB in a **single
   DIMM**, so it is single-channel. Memory-bound decode will be roughly half
   what the same capacity in two sticks would achieve. This is the single
   largest performance constraint on the machine, and it is a hardware
   limitation, not an engine one.
4. VRAM exceeds RAM by a wide margin. The GPU is not the scarce resource; system
   RAM is. This inverts the usual assumption that VRAM is the bottleneck.

## 2. Build environment

| Tool | Status |
|---|---|
| MSVC | **absent** — Build Tools 2022 carries only MSBuild, no VC tools, no Windows SDK |
| GCC | MinGW-w64 16.2.0 (UCRT, POSIX threads), installed per-user via winget |
| CMake | 4.4.4 |
| Ninja | 1.13.2 |
| Clang | absent |
| Python | 3.14 |
| HIP SDK | **absent** |
| Vulkan SDK | **absent** (loader and AMD ICD present; no headers, no `glslc`) |

Decisions taken, recorded in ADR-0001:

- Build with GCC now, keeping the source MSVC-clean.
- Install per-user, because the session is not elevated and a system-wide
  install would block on UAC.
- Link executables statically so they run without MinGW runtime DLLs on `PATH`.
  Without this, the binaries failed at startup with `0xC0000135`.

## 3. GPU compatibility

- The GPU is fully functional for Vulkan: the loader enumerates it and reports
  the expected 15.67 GiB device-local heap.
- **HIP cannot be evaluated yet.** No SDK is installed, and the RX 9060 XT is
  `gfx1200`, new enough that toolkit support must be verified rather than
  assumed. HIP on Windows also requires MSVC, which is absent.
- No GPU code exists in the repository. Nothing here is blocked by a bug; it is
  blocked by two missing installs.

## 4. Memory capacity

At audit time: 15.86 GiB visible, 1.95 GiB free, 87 percent in use.

The budget model (ADR-0003) therefore plans for 1.75 GiB:

```
os_reserve = max(15.86 * 0.25, 2 GiB) = 3.97 GiB
usable     = min(15.86 - 3.97, 1.95 * 0.90) = 1.75 GiB
  weights  1.09 GiB   kv_cache 0.40 GiB
  scratch  0.14 GiB   sessions 0.12 GiB
```

Practical consequence: a 7B Q4 model (about 4.4 GiB of weights plus KV cache)
does not fit on this machine while other applications are running. Even a 3B Q4
model at roughly 1.9 GiB is marginal. Small models (0.5B–1.5B) are the realistic
inference targets until RAM is expanded, and memory-constrained mode is the
default reality rather than an edge case.

## 5. Baseline inference benchmarks

Established in revision 2 (this section originally reported "not established").

llama.cpp was built from source at commit `a91eab6`, GCC 16.2, CPU backend with
AVX2/FMA/F16C and no AVX-512, and run against the models already present on the
machine:

| Model | Size | Prompt (pp512) | Decode (tg128) |
|---|---|---|---|
| qwen2.5-0.5B-Instruct Q4_K_M | 463 MiB | 407.6 ± 29.7 t/s | 8.73 ± 2.30 t/s |
| qwen2.5-1.5B-Instruct Q4_K_M | 1.04 GiB | 207.6 ± 0.7 t/s | 7.22 ± 0.75 t/s |

This is a **reference point, not a comparison**. TGE has no inference path, so
there is nothing to compare. It does establish the two numbers TGE must beat
once milestone 0.2 lands, and it confirms the claim in section 4: decode is
memory-bandwidth-bound. The measured 12.4 GB/s read bandwidth against a 1.04 GiB
model gives a ceiling of roughly 11 tokens/s; llama.cpp achieves 7.2, so it is
already within 35 percent of the hardware wall.

## 6. Initial architecture

Three layers, one-way dependencies:

- `tge::cpu` detects capability and topology from CPUID and
  `GetLogicalProcessorInformationEx`, and filters vector features through XCR0
  so a masked hypervisor cannot produce an illegal instruction.
- `tge` selects kernels once from the filtered feature set. There is no silent
  per-call downgrade.
- `cpu::gemm_avx2` implements a 6x16 microkernel; `tge::ops` holds the reference
  implementations that every optimized kernel is validated against.
- `tge::memory` provides `query_system_memory`, `AlignedBuffer` and
  `MemoryBudget` with four independent pools.

Full description: `docs/architecture.md`.

## 7. Completed implementation

| Component | File | State |
|---|---|---|
| CPUID + XCR0 detection | `src/core/cpu.cpp` | tested |
| Windows topology incl. P/E classes | `src/core/cpu.cpp` | tested |
| Kernel dispatch | `src/core/dispatch.cpp` | tested |
| Work-stealing thread pool | `src/core/thread_pool.cpp` | tested |
| Reference GEMM and vector ops | `src/tensor/reference.cpp` | tested |
| AVX2 + FMA GEMM, packed and blocked | `src/cpu/avx2/gemm_avx2.cpp` | tested for numerical agreement |
| AVX2 + FMA GEMV, threaded decode kernel | `src/cpu/avx2/gemv_avx2.cpp` | tested for numerical agreement |
| Streaming bandwidth kernels | `src/cpu/avx2/mem_bandwidth_avx2.cpp` | tested |
| Scalar streaming fallbacks | `src/cpu/scalar/mem_bandwidth_scalar.cpp` | tested |
| Q8_0 GEMV (decode weights) | `src/cpu/avx2/q8_gemv_avx2.cpp` | tested against scalar |
| GGUF reader (memory mapped) | `src/models/gguf.cpp` | exercised on real models |
| Qwen2 decoder and greedy loop | `src/models/qwen2.cpp` | matches llama.cpp token for token |
| System memory query | `src/memory/system_ram.cpp` | tested |
| Aligned allocator | `src/memory/allocator.cpp` | tested |
| Memory budget manager | `src/memory/budget.cpp` | tested |
| CLI (`info`, `bench gemm`, `bench mem`) | `apps/cli/main.cpp` | exercised |
| Test harness (44 cases) | `tests/` | 44/44 passing |
| Build + gate scripts | `scripts/` | `scripts\test.cmd` passes |

Measured, best observed, five runs per configuration (the machine's background
memory pressure makes single samples unreliable; medians are in
`docs/benchmarks.md`):

| Shape | Before (1 thread) | After (1 thread) | After (16 threads) |
|---|---|---|---|
| 512³ | 88.5 GFLOPS | 75.9 | 120.6 |
| 1024³ | 56.8 GFLOPS | 66.8 | 184.9 |
| 2048³ | 18.2 GFLOPS | 43.3 | 246.5 |

Maximum numerical deviation from the reference across all shapes: 1.9e-04.

The rewrite replaced an unpacked, unblocked, single-threaded kernel with packed
A and B panels, 144/256/256 cache blocking, a first-pass store instead of a
read-modify-write, and dynamic thread work claiming. The 2048³ case, where the
old kernel collapsed to 18 GFLOPS, improved 13.5x.

### The decode primitive and what it says

GEMM is prefill. Decode, the long tail of token generation, is a matrix-vector
product, so a dedicated GEMV kernel was written and measured.

| Matrix | Size | 1 thread | 16 threads |
|---|---|---|---|
| 4096 x 4096 | 64 MiB | 18.4 GB/s | **35.4 GB/s** |
| 8192 x 8192 | 256 MiB | 13.4 GB/s | 28.7 GB/s |

Against a plain threaded read of 256 MiB (25.6-29.4 GB/s), the kernel is at the
hardware wall. Nothing further can be won inside it; only fewer bytes or better
pages would help.

The number that matters: at 28-35 GB/s, a 1 GiB model has a decode ceiling of
27-33 tokens per second. The measured llama.cpp baseline on a 1.04 GiB model is
7.22 tokens per second. **Decode is running at roughly a quarter of available
memory bandwidth**, which means the hardware is not the obstacle and a decoder
built against this foundation has real room to work in.

## 8. Remaining tasks

In the order that matters:

1. **GGUF parsing and a CPU decoder** — the entry point to milestone 0.2, and
   the only way to turn the llama.cpp baseline into an actual comparison.
2. **AVX-VNNI quantized dot product** — quarters the bytes that must cross the
   bus for Q4_K weights, which is the only large lever left on a bandwidth-bound
   decode path.
3. **Large pages** — the 256 MiB GEMV result shows TLB pressure cutting
   bandwidth by 20 percent. `SeLockMemoryPrivilege` is not granted to this
   account, so this cannot be implemented or verified without an administrator
   changing Local Security Policy.
4. **A per-thread scratch arena for packed panels** — packing currently
   allocates on every call.
5. **MSVC install (elevated)** — needed for HIP and for MSVC code-generation
   comparison.
6. **HIP SDK verification for gfx1200** — before any GPU milestone is scheduled.

## 9. Known limitations

- Prompt processing is batched and measured at 170.9 t/s against llama.cpp's
  285.3 on the same weights. The gap is arithmetic: the batch kernel sustains
  about 207 GFLOPS against llama.cpp's 358.
- No tokenizer. Tokens are supplied and returned as ids. This matches how
  `llama-bench` measures and is the reason a speed comparison was possible at
  all, but it is not a usable end-user path yet.
- Sampling supports temperature, top-k, top-p, min-p, repetition penalty and a
  seed. The benchmarks use greedy so their numbers are reproducible.
- Context defaults to 4096 tokens and is selectable with `--context` up to the
  model's own maximum (32768 hard cap), not the 2048 this build used to allocate.
- Quantized formats supported: Q8_0, Q5_0, Q4_0, Q4_K, Q5_K, Q6_K. Validated by
  running a Q4_K_M model, which mixes four of them, and reproducing llama.cpp's
  output token for token.
- Token ids decode to text (`--text`). Encoding text to ids is **not**
  implemented, so prompts are still supplied as ids.
- Three robustness gaps were closed in the GGUF reader: an array length that
  could exhaust memory before it was validated, pointer arithmetic that could
  overflow on a corrupt length field, and row widths that are not whole Q8
  blocks. The parser now has round-trip tests, including the failure paths.
- The AVX2 GEMM is packed, blocked and threaded, but has no per-thread scratch
  reuse. At 512 cubed single-threaded it is about 14 percent slower than the
  simpler kernel it replaced.
- Software prefetch was tried in both the GEMV and the bandwidth path and removed
  in both: the hardware prefetcher already covers a contiguous per-thread stream.
- Large pages are unavailable on this account, so the TLB lever is unverified.
- Every performance number is a GCC build. No MSVC build has been compiled.
- The budget policy fractions (62/23/8/7) are policy guesses, not measurements.
- Benchmark variance on this machine is large (up to 2x single-run swings)
  because other applications hold 3 GB of the 16 GB. All headline numbers are
  medians or best-of-five, and are labelled as such.
- Nothing GPU-related exists.
