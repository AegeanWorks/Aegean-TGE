# MEASURED: llama.cpp re-comparison on the current tree — decode now ahead like-for-like (1.55x tg128, 1.14x tg32), prefill still 1.3-1.5x behind

- **Question**: the 2026-11-08 comparison predates the fused Q4_K/Q6_K decode dots, the per-32 kq8d tiers and the max stack. Re-run the same-window comparison on today's tree, and compare **like-for-like** for the first time: llama.cpp's default Q4_K CPU path quantises activations to Q8_K (lossy), so the fair TGE arm is the priced lossy tier (`--kq8d head,ffn,down,attn`, KL 4.382e-4 / 393/399), not the exact default. Match llama's default f16 KV too.
- **Cycle date**: 2026-11-11
- **Verdict**: **MEASURED.** Same window, Llama-3.2-3B-Instruct-Q4_K_M, 6-thread llama vs TGE:
  - **Decode, like-for-like** (TGE max tier + f16 KV): **23.25/21.88 t/s vs llama 14.58** at pp512/tg128 (**1.55x**), and **17.66/17.52 vs 15.38** at pp2048/tg32 (**1.14x**). TGE's exact default is 0.82x / 0.69x — the tier is what crosses llama.
  - **Prefill** (same KV tier, f16): TGE 69.3/68.6 vs llama 89.76 (**0.77x**), and 53.2/52.7 vs 78.63 (**0.67x**). With f32 KV TGE is 58.4 and 51.8 (0.65-0.66x). The prefill gap survives every arm; llama's Q8_K prefill GEMM is still 1.3-1.5x ahead.
  - Thread-matched arms (`-t 6` both, f32 KV): TGE pp 43.7/42.3, tg 11.16/9.83 -> 0.49/0.54x pp and 0.77/0.64x tg; the best-threaded TGE arms above are the fair product comparison because the tree's default prefill already uses all logical cores (H3).
- **Tree state at close**: tier spike reverted from `src/cpu/avx2/dequant_batch_kq8_avx2.cpp`, `include/tge/cpu/kernels.hpp`, `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp`, `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Setup

| | llama.cpp | TGE |
|---|---|---|
| binary | `llama-bench.exe` build cd8427d | `tge.exe` (HEAD `ab59139`) |
| threads | `-t 6 -ngl 0` | decode 10 (H3 default), prefill 16 logical; matched arm decode/prefill 6 |
| KV | f16 (default) | f32 and f16 arms |
| weights | Q4_K_M, 1.87 GiB | same file |
| prompts | llama-bench internal pp512/pp2048 | repeated fox sentence x51 (510 tok) / x205 (2050 tok) |
| reps | `-r 3` | 2-3 alternating same-window runs |

Prompt-shape caveat (unchanged from the prior report): llama-bench's internal prompt is not our text; token counts match at 512/2048, and only same-window ratios are claimed.

## 2. Same-window results

### llama.cpp (6 threads, f16 KV)

| test | t/s |
|---|---:|
| pp512 | 89.76 ± 1.45 |
| pp2048 | 78.63 ± 0.55 |
| tg128 | 14.58 ± 0.31 |
| tg32 | 15.38 ± 0.09 |

### TGE arms

| arm | pp510/pp2050 t/s | tg128/tg32 t/s | pp ratio (512/2048) | tg ratio (128/32) |
|---|---|---|---|---|
| matched, f32 KV (`--threads 6 --prefill-threads 6`) | 43.7 / 42.3 | 11.16 / 9.83 | 0.49 / 0.54 | 0.77 / 0.64 |
| default, f32 KV (`--threads 10`, prefill 16) | 58.4 / 51.8 | 11.97 / 10.68 | 0.65 / 0.66 | 0.82 / 0.69 |
| default, f16 KV | 69.0 / 52.9 | - | 0.77 / 0.67 | - |
| **max tier, f16 KV** (`--kq8d head,ffn,down,attn`) | **69.3 / 52.9** | **22.57 / 17.59** | 0.77 / 0.67 | **1.55 / 1.14** |

Raw runs: TGE f16+tier 510 tokens prefill 69.3/68.6 tok/s, decode 23.25/21.88; 2050 tokens prefill 53.2/52.7, decode 17.66/17.52. llama numbers are `-r 3` means.

## 3. Reading

- **Decode, like-for-like (both lossy, both f16 KV): TGE is now ahead** — 1.55x at pp512/tg128 and 1.14x at pp2048/tg32. That reverses the 2026-11-08 reading (base/llama 0.56-0.70, stack/llama 0.98-1.06) and is consistent with the fused Q4_K/Q6_K dots (1.3-1.4x kernel) plus the per-32 max tier (1.42x over the exact default).
- **Prefill remains the gap**: 0.77x at 512 and 0.67x at 2048 on matched KV tiers; llama's Q8_K prefill GEMM is 1.3-1.5x faster. This is the same conclusion as 2026-11-08 (llama 1.25-1.6x / 1.6-2.1x), now with a narrower 512 gap because both arms moved to f16 KV. The q4k-r2 report already closed the in-tree prefill levers (AVX2 roofline; VNNI gated by fidelity) — this measurement re-confirms the remaining gap is the activation-quantised GEMM, which TGE ships only behind `--vnni`.
- **Exact default vs llama is still behind on both axes** (0.82x tg, 0.65x pp). The product story is: default = exact; opt-in max tier = llama-class fidelity at llama-beating decode.
- Fidelity note: the tier is priced at mean KL 4.382e-4 / 393/399 on f32 KV; with f16 KV the priced f16-KV error (1.7e-6) is negligible on top. llama's own f32-relative fidelity was not measured here, so "like-for-like" means "both quantise activations to 8-bit blocks", not an equivalent KL claim.

## 4. Exact commands

```text
llama-bench.exe -m Llama-3.2-3B-Instruct-Q4_K_M.gguf -t 6 -ngl 0 -r 3 -p 512 -n 128 -p 2048 -n 32
tge generate --model <gguf> --prompt <fox x51|x205> --n 128|32 --threads 6 --prefill-threads 6 --gpu off
tge generate ... --threads 10 --gpu off
tge generate ... --threads 10 --kv f16 --gpu off
tge generate ... --threads 10 --kv f16 --gpu off --kq8d head,ffn,down,attn
```

## 5. Reimplementation recipe

1. Re-apply the per-32 max tier (`reports/idea-invented-kq8d-full-stack-per32-keep.md` §8) so `--kq8d head,ffn,down,attn` exists; build + 171/171.
2. Run llama-bench as above (3 reps) and the four TGE arms alternating within the same window; compare only same-window ratios.
3. Expect decode like-for-like 1.1-1.6x ahead, prefill 0.65-0.8x behind. A future prefill win needs the VNNI-class integer GEMM (`--vnni`, KL 1.5e-3, ids flip on 30/1023) or a cheaper exact GEMM, not another decode tier.
4. Revert the tier spike (report-only).

## 6. Exact results summary

| axis | TGE exact | TGE max tier + f16 KV | llama default | like-for-like verdict |
|---|---:|---:|---:|---|
| pp512 | 58.4 | 69.3 | 89.76 | llama 1.30x ahead |
| pp2048 | 51.8 | 52.9 | 78.63 | llama 1.49x ahead |
| tg128 (512 ctx) | 11.97 | 22.57 | 14.58 | **TGE 1.55x ahead** |
| tg32 (2048 ctx) | 10.68 | 17.59 | 15.38 | **TGE 1.14x ahead** |

No push, no merge, no force-push. The tier exists only in the rebuilt binary; the committed tree is unchanged except this report.
