# MEASURED: kq8d max tier under speculation at 2k — neutral-to-negative (median 0.966x), confirming they are substitutes at long context too

- **Question**: The stacking test (`idea-invented-kq8d-speculation-stacking-measured.md`) showed spec and the per-32 kq8d tier are substitutes at a 1,200-token prompt: the verifier dominates a full-accept step and the tier only touches 1-row corrections. At 2k the attention share rises (21% of decode) and the verifier is heavier; does the tier become worth anything there, and does it disturb the ids?
- **Cycle date**: 2026-11-11
- **Verdict**: **MEASURED — no, it stays neutral-to-negative.** 2,050-token periodic prompt, `--context 2560 --n 32 --kv f16 --threads 10 --gpu off`, greedy, spec on:
  - off: 31.50/31.86/30.66 t/s; tier (`--kq8d head,ffn,down,attn`): 30.27/32.12/29.61 t/s -> ratios **0.961/1.008/0.966**, **median 0.966x** (1/3 positive, within box noise but no gain in any meaningful sense).
  - **ids identical** (`4C-1C-DE…`) in all six runs; spec line identical (6 steps, 5.17 accepted/step).
  - This extends the earlier substitute finding to long context: at 2k the tier is still dead weight under speculation, because the 6-row verifier runs the kq8 per-256 batch kernel regardless of the kq8d scopes (the scopes only reach the 1-row correction forward).
- **Tree state at close**: tier spike reverted from `src/cpu/avx2/dequant_batch_kq8_avx2.cpp`, `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp`, `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

- Speculation's steady state: 6-row verify via `prefill_run` -> `batch_matvec` with `float_only = false` -> the kq8 per-256 batch kernel (occupancy-invariant, near-exact), then a 1-row correction via `forward()` -> `matvec` (where the kq8d scopes live).
- So a full-accept step is ~6/7 verifier (unaffected by kq8d) and ~1/7 correction (affected). At 2k the attention inside the verify grows, which increases the verifier's share further and shrinks the tier's already small slice.
- Prediction therefore: tier/spec ~= 1 - epsilon, with epsilon growing with context. Measured 0.97-1.01.

## 2. Falsifier (stated before the run)

**KEEP (as a recommended combination)** if the tier adds >= 10% to the spec run at 2k with identical ids; **MEASURE/KILL the combination** if neutral or negative.

Measured median 0.966x, ids identical -> the combination is not recommended; the tier remains a plain-decode lever.

## 3. File anchors (current HEAD, read-before-write verified)

- `src/models/qwen2.cpp:1125-1142` — `batch_matvec` and the verifier's kq8 path.
- `src/models/qwen2.cpp:2713-2727` — `verify_draft_logits` (6-row verify).
- `apps/cli/main.cpp:788-922` — fused greedy spec steady state (correction forward).
- `src/models/qwen2.cpp:860-861`, `:2178-2180`, `:2218-2219`, `:2228-2229`, `:2233-2234` — the kq8d 1-row call sites.
- `reports/idea-invented-kq8d-speculation-stacking-measured.md` — the 1,200-token result this extends.
- `reports/idea-invented-spec-at-2k-measured.md` — the 2k spec baseline (2.74x over off) used here.

## 4. Measured numbers

| pair | spec off t/s | spec + tier t/s | ratio | ids |
|---:|---:|---:|---:|---|
| 1 | 31.50 | 30.27 | 0.961 | identical |
| 2 | 31.86 | 32.12 | 1.008 | identical |
| 3 | 30.66 | 29.61 | 0.966 | identical |
| **median** | 31.50 | 30.27 | **0.966** | 3/3 |

Spec line in every run: 6 steps, 31 draft tokens accepted (5.17/step), 36 tokens emitted.

## 5. Reading

- The tier and speculation remain **substitutes** at every context measured (1.2k and 2k): use `--speculative` on repetitive/templated prompts (2.2-2.8x at 2k) and `--kq8d head,ffn,down,attn` on plain decode (1.42x at short context). Combining them costs ~3% at 2k and gains nothing.
- The only way they could compose is a verify path that uses the same per-32 integer arithmetic as the tier (so both rows share it) — and that was already tried and killed as speed-neutral (`idea-invented-kq8d-per32-verify-killed.md`), because the verifier is weight-stream-bound in either integer form.
- `--speculative auto`'s probation already handles the case where drafts fail (it bails, leaving the tier in charge); that remains the correct automatic policy.

## 6. Reimplementation recipe

1. Re-apply the per-32 max tier (`reports/idea-invented-kq8d-full-stack-per32-keep.md` §8) and build + 171/171.
2. Run `generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt "<fox x205>" --context 2560 --n 32 --kv f16 --threads 10 --gpu off --print-ids --speculative [--kq8d head,ffn,down,attn]`, alternating pairs.
3. Expect off 30-32 t/s, tier 29.6-32.1 t/s, ratios 0.96-1.01, ids identical. Revert; no code ships.

## 7. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge generate ... --prompt "<fox x205>" --context 2560 --n 32 --kv f16 --threads 10 \
  --gpu off --print-ids --speculative [--kq8d head,ffn,down,attn]  -> §4 table
git checkout -- src/cpu/avx2/dequant_batch_kq8_avx2.cpp include/tge/models/qwen2.hpp src/models/qwen2.cpp apps/cli/main.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the rebuilt binary; the committed tree is unchanged except this report.
