# INVENTED: stacking the per-32 kq8d full stack with prompt-lookup speculation — MEASURED (MIXED: -6% where drafting works, +51-56% where it fails; verifier is float-only)

- **Idea**: The per-32 kq8d full stack (`--kq8d head,ffn,down,attn`) gives ~1.42x on plain decode; speculation gives 3-4x on repetitive prompts. The obvious question: do they multiply? Price `--speculative` with and without the tier on two workload regimes — a periodic prompt (drafts accept 5.09/step, so the 6-row verifier dominates) and free prose (drafts accept 1.00/step, so almost every forward is a 1-row correction).
- **Cycle date**: 2026-11-11
- **Verdict**: **MEASURED — they do not stack; they are substitutes.** Periodic + spec: the full stack is **0.909-0.950x** (median ~0.94x) — slightly *slower*. Prose + spec: the full stack is **1.561x / 1.513x** — near its no-spec 1.42x, because the 1-row correction forwards use kq8d.
  The cause is design, not noise: the verifier already uses the kq8 per-256 integer batch kernel, so the kq8d scopes only touch the 1-row corrections and the 6-row verify cannot gain from them. Where drafting works, the run is ~all verifier (float) and the tier is dead weight; where drafting fails, the run is ~all 1-row correction forwards and the tier pays in full.
  Ids: periodic identical with and without the tier (`9D-93-29…`); prose flips with the tier (`AD-3D-F2…` vs `97-FD-AF…`) — the tier's priced near-tie behaviour, unchanged by speculation.
- **Correction (2026-11-11, follow-up cycle)**: the mechanism claims elsewhere in this report say the verifier is float-only. That is wrong: `prefill_run` calls `batch_matvec` with `float_only = false`, so verify already uses the kq8 per-256 integer batch kernel. Only `forward_batch` (continuous mixed batches) forces float, for the occupancy invariant. The measured numbers stand; the cause is verify being already integer and memory-bandwidth-bound. See `reports/idea-invented-kq8d-per32-verify-killed.md`; the per-32 batch verifier is speed-neutral (0.97-1.06x).
- **Tree state at close**: spike reverted from `src/cpu/avx2/dequant_batch_kq8_avx2.cpp`, `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp`, `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism (current tree, read before measuring)

- Speculation: `verify_draft` / `verify_draft_logits` run `prefill_run` on the 6-row feed through `batch_matvec` with `float_only = false`, so every K-quant matvec runs the shipped kq8 per-256 integer batch kernel (`gemv_dequant_batch_kq8_avx2`). The `float_only = true` calls in `src/models/qwen2.cpp:1132-1141` belong to `forward_batch` (continuous mixed batches), not to verify; the kq8d scopes therefore cannot change verify arithmetic at all.
- Decode corrections (the 1-row forwards for the bonus token and after a rejection) go through `forward()`/`matvec()`, which honour the kq8d scopes. In the fused steady state with 5.09 accepted/step, ~5 of every 6 rows are verify and one is a 1-row correction; in prose, drafts fail, the fused path exits after 2 steps, and the rest is 1-row decode.
- The spike re-applied the per-32 kernel (both K-quants) and the `ffn`/`head,ffn,down`/`head,ffn,down,attn` scope wiring from the last three reports, so the tier is exactly the one those reports keep.

## 2. Predicted gain (with the math)

If the verify is 6 rows of already-integer batch math and a correction is 1 row, a full-accept step costs roughly 6 integer row-equivalents + 1 integer row-equivalent; the tier can only accelerate the last ~1/7 of the work. With kq8d ~1.5x faster on that 1/7, the step-time ratio is `(6 + 1/1.5)/(6+1) = 0.976` — a ~2% *loss* is even possible from the extra quantisation. Measured -6%.

Where drafts fail every step is a 1-row forward: 100% of the work is on the tier path, so the run should match the no-spec tier (~1.42x). Measured 1.51-1.56x (the spec path's 2 warm-up steps are also 1-row).

## 3. Falsifier (stated before measuring)

**KEEP (as a recommended combination)** if the tier adds >= 10% to a speculation run on the repetitive prompt with ids identical; **MEASURE/KILL the combination** if it is neutral or negative there even though it pays where drafts fail.

Periodic + spec measured 0.909-0.950x -> combination is not recommended; prose + spec measured 1.51-1.56x -> the tier is still usable under forced speculation on non-repetitive prompts.

## 4. File anchors (current HEAD, read-before-write verified)

- `src/models/qwen2.cpp:1125-1142` — `batch_matvec` routing and the `float_only` guard (the reason the verifier cannot use kq8d).
- `src/models/qwen2.cpp:2713-2727` — `verify_draft_logits` (6-row `prefill_run`).
- `apps/cli/main.cpp:788-922` — fused speculative steady state (accepted/step, correction forward); `:1081-1210` — sampled speculative path.
- `src/models/qwen2.cpp:860-861` (head), `:2178-2180`/`:2218-2219` (q/k/v/o), `:2228-2229` (gate/up), `:2233-2234` (down) — the 1-row path's kq8d call sites.
- `reports/idea-invented-kq8d-full-stack-per32-keep.md` — the tier being stacked (1.42x, 4.382e-4 / 393/399).
- `reports/idea-invented-spec12-wide-verify-killed.md`, `reports/idea-invented-speculation-auto-probation-keep.md` — verify-cost and spec-regime precedents.

## 5. What was done (temporary spike, fully reverted)

Re-applied the per-32 kernel (Q8K32Block, quantize_row_q8k32, unpack_q4k_scale_min, dot_q4k_q8k32, dot_q6k_q8k32, row-kernel branch) and the `ffn` + combined scope wiring, built, 171/171. Ran `--speculative` with and without `--kq8d head,ffn,down,attn` on the periodic prompt (3 rotated pairs) and the prose prompt (2 rotated pairs), n=64, threads 10, f32, `--print-ids`. Reverted four files; rebuilt; 171/171; gates 4/4; clean tree.

## 6. Measured numbers

`Llama-3.2-3B-Instruct-Q4_K_M.gguf`, `--n 64 --kv f32 --threads 10 --gpu off --speculative --print-ids --kq8d <scope>`.

### Periodic prompt (1,200 tokens; 11 steps, 5.09 accepted/step)

| pair | off t/s | full t/s | full/off | ids |
|---:|---:|---:|---:|---|
| 1 (off first) | 40.49 | 38.45 | 0.950 | identical |
| 2 (full first) | 38.28 | 34.78 | 0.909 | identical |
| 3 (off first) | 37.81 | 35.67 | 0.943 | identical |
| **median** | 38.28 | 35.67 | **0.943** | 3/3 |

The tier is a 4-6% *loss* in that build: the verifier (6 integer batch rows) dominates and the 1-row correction is a seventh of the work; the extra per-32 quantisation and int-to-float work in that seventh can even lose a little. The follow-up cycle's head,down reading (0.96-1.05 under spec) shows the sign is not stable; read the combination as neutral.

### Prose prompt (24 tokens; 2 steps, 1.00 accepted/step, then baseline decode)

| pair | off t/s | full t/s | full/off | ids |
|---:|---:|---:|---:|---|
| 1 | 12.93 | 20.18 | 1.561 | flips (`AD-3D-F2`, expected) |
| 2 | 13.42 | 20.30 | 1.513 | flips |
| **median** | 13.18 | 20.24 | **1.537** | |

Near the no-spec 1.42x (the spec warm-up steps add a little), because every post-warm-up forward is a 1-row decode that honours the tier.

## 7. Consequence for usage and for a future idea

- **Usage**: pick one, not both. On repetitive/templated work, prompt-lookup speculation (3-4x) dwarfs the tier (1.4x) and combining *loses* ~5%. On free prose, speculation does nothing; the tier gives 1.4-1.5x, and `--speculative auto` (the probation mode) will bail out on its own and leave the tier in charge. That is the correct combination story: auto-spec + tier is fine because auto disables spec where it would not pay.
- **Future idea (not attempted)**: make the verifier use a per-32 batch kernel (`gemv_dequant_batch_kq8_32_avx2`, M=6) that is bit-identical per row to `dot_q4k_q8k32`/`dot_q6k_q8k32`. Then verify and decode share arithmetic and the tier would compose with speculation — but that also removes the current `float_only` occupancy-invariant guarantee for near ties between 1-row and 6-row math, so it needs its own exactness proof first. Recorded here so it is not re-discovered as a surprise.
- The ids caveat stands: the tier flips near ties on prose (max tier matched off on the dragon prompt in the earlier cycle; the full stack does not). Auto-spec does not change that.

## 8. Reimplementation recipe (re-run this measurement)

1. Apply `reports/idea-invented-kq8d-per32-q6k-max-tier-keep.md` §8 (kernel) and `reports/idea-invented-kq8d-full-stack-per32-keep.md` §8 step 3 (scope wiring incl. `head,ffn,down,attn`).
2. Build + `tge_tests` 171/171.
3. Periodic: `tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt <fox x120> --n 64 --kv f32 --threads 10 --gpu off --speculative --print-ids [--kq8d head,ffn,down,attn]` in rotated off/full order, 3+ pairs -> expect 0.91-0.95x.
4. Prose: same with the dragon prompt -> expect 1.5-1.6x and an id flip.
5. Revert. The finding is the interaction, not a code change.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge generate ... --speculative --print-ids [--kq8d head,ffn,down,attn]
  periodic, 3 rotated pairs -> 0.950, 0.909, 0.943 (median 0.943), ids identical
  prose,    2 rotated pairs -> 1.561, 1.513 (median 1.537), ids flip as priced
git checkout -- src/cpu/avx2/dequant_batch_kq8_avx2.cpp include/tge/models/qwen2.hpp \
  src/models/qwen2.cpp apps/cli/main.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the rebuilt binary; the committed tree is unchanged except this report.
