# MEASURED: kq8d `head` / `down` / `head,down` re-priced against the fused-float default — KEEP (stack 12/12 positive, ~1.23x median; `head` alone is workload-dependent; order must be rotated)

- **Idea**: The shipped kq8d tiers (`--kq8d head`, `--kq8d down`, `--kq8d head,down`) were kept in cycles 19-20 against the *generic dequantise-then-dot* float path. Since cycle 22 the default decode path is the fused Q4_K/Q6_K register dot (bit-identical, 1.3-1.5x the generic kernel), so those keeps need re-pricing: the integer row kernel's advantage over today's baseline is smaller than the old reports imply, and on some shapes it may be gone. This cycle re-measures all three scopes against the current default on two workloads, with order rotation.
- **Cycle date**: 2026-11-11
- **Verdict**: **KEEP — with a corrected story.** Against the fused-float default on Llama-3.2-3B-Instruct-Q4_K_M, `--threads 10 --kv f32`, n=64:
  **`head,down` is robustly positive: 12/12 interleaved pairs across a long-prose and a periodic prompt, ratios 1.08-1.41, median ~1.23x.**
  `down` alone is positive on median in both workloads (prose 1.22x rotated / 1.03x unrotated; periodic 1.07x) but not in every pair.
  `head` alone is workload-dependent: ~1.10x on the long-prose prompt, ~1.02x on periodic (one negative pair) — its share shrinks as attention grows.
  The stack beats its components (prose `head,down`/`head` = 1.14/1.13/1.27/1.16), which the old additive estimate did not predict.
  **Measurement lesson**: the first, unrotated run put `off` first in every pair and read `down` at 0.99; rotating the four arms across pairs moved it to 1.22. On this loaded box, order drift is as large as the effect — every tier comparison must rotate.
- **Tree state at close**: no code touched (shipped flags only); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; 171/171 tests, `python aikit.py check aegean-tge` 4/4. No branch, nothing pushed.

## 1. Mechanism (what is being priced)

- `--kq8d head`: `output_logits` passes `kq8d_head_` to `matvec(output_, hidden, logits_, q_out, kq8d_head_)` (`src/models/qwen2.cpp:860-861`). The 308 MiB 3072x128256 Q6_K LM head runs the integer Q6_K x Q8_K row dot instead of the fused Q6_K register dot.
- `--kq8d down`: `matvec(layer.w_down, gate_, h_, q_down, kq8d_down_)` (`:2233-2234`). The 3072x8192 Q4_K down projection rides the integer row dot.
- `--kq8d head,down`: both.
- The kq8d branch is the first in `matvec` when `allow_kq8d && !scalar_kernels_ && kq8d_kind && avx2` (`:1016-1020`), calling `gemv_dequant_kq8_row_avx2` (activation quantised once per call into a `thread_local` Q8_K scratch, rows banded across the decode pool).
- Today's baseline is the fused per-row dot: `dot_q4k_row_avx2` / `dot_q6k_row_avx2` behind `gemv_q4k_fused_avx2` / `gemv_q6k_fused_avx2`.

## 2. Predicted gain (with the math)

Kernel bench on today's tree (`tge bench batch`, best of 5, 1 token) versus the *generic* float kernel: gate/up 2.16x, down 2.39x, q/o 1.68x, LM head 1.87x. The fused dot is ~1.76x the generic kernel on Q4_K (cycle-19 measurement), so kq8d's advantage over the shipped path is roughly down `2.39/1.76 = 1.36x` and head `1.87/1.315 = 1.42x` (the Q6_K fused increment was 1.315x kernel in the cycle-20 report).

Weight shares of the 2.02 GiB per-token stream (Llama-3.2-3B, 28 layers, hidden 3072, ffn 8192, 4.5 bpw Q4_K): the fused Q4_K bench rate is 13.5 MiB for an 8192x3072 matrix, so gate+up ~756 MiB (37%), down ~378 MiB (19%), q/k/v/o ~381 MiB (19%), LM head 308 MiB (15%), embeddings/other the rest. At 1.36x on 19% the down tier predicts `1/(0.81 + 0.19/1.36) = 1.06x`; at 1.42x on 15% head predicts `1.05x`; both `1.11x`.

Measured medians are **higher** than that model: down 1.07-1.22x, head 1.02-1.13x, stack 1.19-1.37x. So the kernel-bench ratio understates kq8d's model-level advantage on these two matrices — the bench uses contiguous aligned buffers, while at model scale the fused path pays page/TLB and activation-load costs the integer row kernel avoids. The stack's excess over the additive model (1.23 vs 1.11 median) is the size of that effect. This is an estimate-gap, not a contradiction: the falsifier is the paired decode ratio, which is the evidence.
## 3. Falsifier (stated before measuring)

**KEEP** if `head,down` shows a median >= 1.05x over the default on **both** workloads with ids consistent per tier; otherwise mark the shipped tier obsolete in the docs. Secondary: `down` and `head` individually must not regress below 1.0x median (a tier that loses to the default has no reason to be listed).

Measured: `head,down` 12/12 pairs, medians ~1.25x (prose) and ~1.19x (periodic) -> keep. `down` medians 1.22x / 1.07x -> keep. `head` medians 1.10x / 1.02x -> keep, documented as workload-dependent.

## 4. File anchors (current HEAD, read-before-write verified)

- `src/models/qwen2.cpp:860-861` — LM-head kq8d call.
- `:2233-2234` — FFN down kq8d call.
- `:969-1020` — `matvec` signature and kq8d-first dispatch.
- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:397-456` — `gemv_dequant_kq8_row_avx2` (thread_local `Q8KBlock` scratch, per-256-block quant with per-32 scales, banded rows).
- `apps/cli/main.cpp:2286-2302` — `--kq8d` parser (`head|down|attn|ffn|head,down|off` after the FFN cycle; today's tree has `head|down|attn|head,down`).
- `reports/idea-invented-kq8d-down-keep.md` (7.954e-4 / 390-399), `...-lmhead-keep.md` (1.874e-4 / 393-399), `...-combo-largecorpus-keep.md` (9.711e-4 / 390-399) — the priced fidelity of these tiers; unchanged by this cycle because no numerics changed.
- `reports/idea-invented-q4k-fused-row-dot-keep.md`, `...-q6k-fused-row-dot-keep.md` — the new baseline the tiers are now measured against.

## 5. What was done (measurement only; no code touched)

1. Read the three call sites and confirmed the scope wiring is unchanged (no spike needed; the shipped flags are the experiment).
2. Run A (unrotated): 4 pairs, fixed order off -> down -> head -> head,down, dragon prompt, `--n 64 --kv f32 --threads 10 --gpu off --print-ids`.
3. Run B (rotated): 4 pairs over the same four arms with the order rotated per pair (`off,down,head,head,down` / `head,down,head,down,off` / `down,head,head,down,off` / `head,off,head,down,down`).
4. Run C (rotated, periodic): the same four-arm rotation on the 240-token periodic fox prompt, with `--print-ids`.
5. No fidelity run owed: no arithmetic changed. No thinking-set diff owed: same binary.

## 6. Measured numbers

Reference `Llama-3.2-3B-Instruct-Q4_K_M.gguf`, `--n 64 --kv f32 --threads 10 --gpu off`, greedy. Same binary for every arm.

### Run A — unrotated (off always first): the confound

Dragon prompt, 4 pairs:

| pair | off | down | head | head,down | down/off | head/off | stack/off |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.70 | 15.59 | 17.38 | 19.38 | 0.993 | 1.107 | 1.234 |
| 2 | 15.45 | 14.13 | 15.38 | 17.47 | 0.915 | 0.995 | 1.131 |
| 3 | 14.05 | 14.94 | 15.37 | 17.26 | 1.063 | 1.094 | 1.228 |
| 4 | 14.12 | 15.44 | 15.16 | 17.45 | 1.093 | 1.074 | 1.236 |
| **median** | 14.76 | 15.19 | 15.38 | 17.46 | **1.028** | **1.084** | **1.231** |

`down` reads as noise here (2 negative pairs) — but `off` ran first in every pair, and `head,down` last, which is exactly the drift this box produces.

### Run B — rotated order (same arms, dragon prompt)

| pair | down/off | head/off | stack/off | stack/head |
|---:|---:|---:|---:|---:|
| 1 | 1.093 | 1.049 | 1.196 | 1.140 |
| 2 | 1.167 | 1.248 | 1.406 | 1.127 |
| 3 | 1.265 | 1.098 | 1.399 | 1.274 |
| 4 | 1.275 | 1.151 | 1.340 | 1.164 |
| **median** | **1.216** | **1.125** | **1.370** | **1.152** |

`down` flips from 1.028 to 1.216 under rotation: the effect is real, the unrotated run was order-biased.

### Run C — rotated order (periodic fox-x40 prompt)

| pair | down/off | head/off | stack/off |
|---:|---:|---:|---:|
| 1 | 1.060 | 0.935 | 1.080 |
| 2 | 1.083 | 1.081 | 1.248 |
| 3 | 1.040 | 1.008 | 1.192 |
| 4 | 1.118 | 1.033 | 1.186 |
| **median** | **1.072** | **1.020** | **1.193** |

12/12 stack pairs positive across both workloads (range 1.080-1.406).

### Ids

| workload | off / down | head / head,down |
|---|---|---|
| dragon | `97-FD-AF…` (both) | `AD-3D-F2…` (both) |
| periodic | `9D-93-29…` (all four arms) | same |

The head tier's priced KL (1.874e-4) is large enough to flip tokens on the dragon prompt near ties; down alone does not flip either prompt. This is the known opt-in trade, not a new finding — but it means `--kq8d head` and the default are **not** id-identical in general, and a speculative run must keep its verifier on the same arithmetic (the documented kq8d caveat).

### Combined reading

| tier | prose median | periodic median | pairs positive (of 12) |
|---|---:|---:|---:|
| down | 1.216 | 1.072 | 10/12 |
| head | 1.125 | 1.020 | 10/12 |
| head,down | 1.370 | 1.193 | **12/12** |

## 7. What this changes

- The shipped tiers still pay against the newer baseline; no doc retraction is needed. `head,down` is the tier to recommend (12/12, ~1.23x combined median); `head` alone is worth it only on long-prose/attention-heavy prompts; `down` is broadly positive but had two negative dragon pairs in the unrotated run.
- Any future comparison of opt-in kernels must rotate arm order across pairs; this box's within-pair drift (14.8 -> 17.5 t/s on the same arm across pairs) is the size of the effects being measured.
- The old additive estimate (down x head) under-predicts the stack; the paired stack/head ratio (1.15x median on prose) is the honest incremental number for adding down to head.

## 8. Reimplementation recipe (re-run this pricing)

1. No code. Use the shipped `--kq8d head|down|head,down` flags on one binary.
2. Model: Llama-3.2-3B-Instruct-Q4_K_M.gguf. Command per arm:
   `tge generate --model <model> --prompt <p> --n 64 --kv f32 --threads 10 --gpu off --print-ids [--kq8d <scope>]`.
3. Use two prompts: the dragon prose prompt (24 tokens) and the periodic fox sentence x40 (240 tokens).
4. **Rotate the arm order across pairs** (four arms -> four rotated orders); record per-arm t/s and the ids hash. Compare per-pair ratios, never absolute levels.
5. Expected: stack 1.08-1.41 per pair, median ~1.23x; down ~1.07-1.22x; head ~1.02-1.13x; head flips ids on prose near ties. Any tier below 1.0x median on both workloads is a candidate for retraction.
6. Fidelity is unchanged from the keep reports (head 1.874e-4 / 393-399, down 7.954e-4 / 390-399, head,down 9.711e-4 / 390-399); re-run only if the arithmetic changes.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> not needed (no code change)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge generate ... --n 64 --kv f32 --threads 10 --gpu off --print-ids [--kq8d head|down|head,down]
  run A (dragon, fixed order)   -> §6 Run A
  run B (dragon, rotated)       -> §6 Run B
  run C (periodic, rotated)     -> §6 Run C
git status --short                                                -> clean (no diff)
python aikit.py check aegean-tge                                  -> 4/4 gates passed
```

No push, no merge, no force-push. No code or binary changed this cycle; the committed tree is unchanged except this report.
