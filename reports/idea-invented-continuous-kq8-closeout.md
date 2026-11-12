# CLOSEOUT: continuous-batch kq8 decode (path-invariant) — one decision matrix over six cycles; promote behind a flag, not by default

- **Scope**: `idea-invented-continuous-kq8-batch-keep.md`, `...-leak-audit-measured.md`, `...-path-invariant-keep.md`, `...-generate-parity-measured.md`, `...-3b-measured.md`, `...-longprompt-measured.md` are six reports on one change. This closeout collapses them into a single decision matrix, names the exact precondition, and gives the promotion checklist. No new mechanism.
- **Cycle date**: 2026-11-11

## 1. The change (final form)

Two edits, which must travel together:

1. `src/models/qwen2.cpp` `batch_matvec`: kq8 K-quant branch condition `tokens > 1` -> `tokens >= 1`.
2. Eight `batch_matvec(...)` calls in `forward_batch` / `output_logits_batch` (`:1703-1828`, `:2039`): `float_only = true` -> `false`.

Edit 2 alone **fails** the occupancy audit (LEAK reproduced: kq8 solo `35-E5-78…` vs busy `99-22-12…`, first divergence token 11/15 at prompt-tokens 24, while the float build is identical). Edit 1 alone changes nothing (occupancy 1 still takes the float kernel). Together they are path-invariant by construction: every occupancy in the batch path runs `gemv_dequant_batch_kq8_avx2`, whose per-row Q8_K quantisation makes a row's result independent of batch composition.

## 2. Decision matrix (all measured, same-window A/B)

### Reference model: Llama-3.2-3B-Instruct-Q4_K_M

| slots | requests x tokens | prompt tokens | float t/s | path-invariant t/s | ratio | evidence |
|---:|---|---:|---|---:|---:|---|
| 4 | 8 x 16 | 24 | 20.19/17.42/17.40/17.02 | 30.75/29.86/28.45/28.90 | **1.523/1.714/1.635/1.698** | 3B report |
| 8 | 12 x 8 | 24 | 22.14/18.46 | 29.84/29.42 | **1.348/1.594** | 3B report |
| 6 | 6 x 8 | 56 | 13.91/13.94/12.21 | 22.01/19.64/19.17 | **1.582/1.409/1.570** | longprompt report |
| 8 | 8 x 8 | 56 | 16.36/16.00 | 25.02/25.24 | **1.529/1.578** | this closeout |

Median across 11 pairs: **~1.57x**; 11/11 positive; range 1.35-1.71.

### Secondary model: qwen2.5-0.5b-instruct-q4_k_m

| slots | requests x tokens | prompt tokens | ratios (12 pairs) | median |
|---:|---|---:|---|---:|
| 4 | 8 x 32 | 8 | 1.263/0.789/1.163 | 1.16 |
| 8 | 16 x 32 | 8 | 1.321/0.954/0.969 | 0.97 |
| 4 | 8/16 x 32 | 8 | 1.033/1.020/1.079/0.944/0.979/1.061 | 1.05 |

Median ~1.06x, 8/12 positive; the box cannot resolve small effects, but no regression beyond noise.

### Parity evidence (the gates that matter more than throughput)

| check | result |
|---|---|
| occupancy audit, 0.5B, prompt-tokens 8/24/48 | pass 3/3 (pt=24 was the failing shape pre-fix) |
| occupancy audit, 3B, prompt-tokens 24/48 | pass 2/2 |
| occupancy audit, 3B, prompt-tokens 56, 6 and 8 slots | pass 2/2 |
| in-run `state: independent (all transcripts identical)` | every run of every arm |
| generate thinking set, 0.5B (5 prompts, greedy n=64) | identical 5/5 |
| generate parity, 3B (n=16 greedy) | identical (`D0-25-F3…`) |
| 20-token and 2,000-token generate prefill timings | ids identical, timing within noise |
| `tge_tests` | 171/171 after every spike and after revert |
| numerically | kq8 is the default prefill kernel, priced 9.248e-12 KL (near-exact); batch outputs change vs the float batch kernel on some shapes, as expected, not a new approximation |

## 3. Verdict

**KEEP-GRADE — promote behind a flag (`--kq8-batch`), not by default in the same step.** The throughput case on the reference model is strong (1.35-1.71x, 11/11), the parity case is complete (audits, generate ids, thinking set), and the only reason not to flip the default is governance: continuous-batch outputs change for existing users, and the shipped `float_only` comment records a past LEAK incident. A flag lets users opt in while the default keeps today's exact float-batch arithmetic; a later default flip is a product decision with this matrix as its evidence.

If the default is flipped, the comment at `src/models/qwen2.cpp:1131-1140` must be rewritten to record that path invariance is structural via `tokens >= 1`, and the regression set is: solo-vs-busy audit (all shapes), generate parity, `tge_tests` 171/171.

## 4. What NOT to redo

- Do not flip `float_only` without `tokens >= 1` — the audit fails at pt=24.
- Do not "fix" the verify path to use the per-32 tier under speculation: it is speed-neutral (`idea-invented-kq8d-per32-verify-killed.md`) and the tier is a plain-decode lever (spec/tier are substitutes at 1.2k and 2k).
- Do not chase the remaining prefill gap with kq8 loop tweaks: bands, two-row sharing, split adds, and both VNNI forms are measured dead ends; the gap is structural.
- Do not add per-32 activation quantisation to the prefill batch kernel for speed (activation-scaled VNNI measured 0.62-0.69x).

## 5. Promotion checklist (for the human gate)

1. `git status` clean at `25c0970` or later; `tge_tests` 171/171; `aikit.py check aegean-tge` 4/4.
2. Re-apply the two edits; rebuild; re-run the audit on 0.5B (8/24/48) and 3B (24/56) — all must match.
3. Re-run the 3B throughput matrix above; require median >= 1.3x at 4/6/8 slots.
4. Run the five-prompt thinking set on 0.5B and the 3B generate parity check.
5. Add `--kq8-batch` (default off) OR flip the default with the comment rewrite; commit with the matrix in the message; push only with explicit approval.
6. After landing, update `docs/measurement-log.md` and the README continuous-batching row with the final numbers.

## 6. Exact commands run and results (this closeout)

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge bench continuous ... --requests 8 --tokens 8 --threads 10 --sequences 8 \
  --kv f16 --prompt-tokens 56                                     -> 16.36/16.00 vs 25.02/25.24 (1.529/1.578)
tge bench continuous ... --requests {1,8} ... --prompt-tokens 56   -> audit 45-B1-96 in all four runs
git checkout -- src/models/qwen2.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. Baseline binary lives at `build/idea/tge-cont-float-base.exe`; the committed tree is unchanged except this report.
