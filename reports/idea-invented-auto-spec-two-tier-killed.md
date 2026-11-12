# MEASURED: two-tier long-prompt auto-spec probe (gate + 1-2 step model probe) — KILLED; the free gate stays, and the blind-spot premise is corrected

- **Idea**: `idea-invented-auto-gate-blindspot-measured.md` claimed the prompt-only lookup gate loses 1.88x on a late-drafting code prompt and prescribed a second tier: when the gate misses, start speculative and run a 1-iteration model probe, keeping spec only if that step accepts a draft. This cycle implements that tier (`auto_probe_steps = gate_hit ? huge : 1`, then `0`) and measures it on periodic, prose and code workloads.
- **Cycle date**: 2026-11-11
- **Verdict**: **KILLED — the probe does not beat gate-off on prose, and the blind-spot premise was wrong.**
  - **Gate-hit workloads pay nothing and keep their win**: periodic 2,050 tokens -> 30.88 vs 12.24 t/s (**2.52x**), gate hit, probe skipped; the 1,530-token repetitive code prompt -> **gate hit** too (22.10 vs 12.38 t/s, 1.79x), and a second code prompt with a random tail (1,543 tokens) also gate-hit (20.04 vs 10.74 t/s, 1.87x). The last five tokens tokenize into pieces that appear earlier even with a "unique" textual tail.
  - **Gate-miss prose pays for the probe and gains nothing**: 1,643-token prose, probe budget 1 (bail after 2 iterations, 0.50 tokens/step) -> 12.69 vs 13.48 t/s (**0.94**, ~6% cost); probe budget 0 (bail after 1 iteration, 0.00 tokens/step) -> **0.90/0.88** (10-12% cost), worse because the first iteration is a full 6-row verify that accepts nothing and emits nothing. Neither beats the gate's plain 0% cost on prose.
  - **Correction**: the blind-spot report's 1.88x figure is real for **forced** `--speculative`, but its premise that the lookup gate would reject that prompt is **false in measurement** — the gate hits it. The blind spot is therefore unproven on the tested shapes; a prompt whose last five tokens are truly absent earlier AND whose continuation returns to a repeated body was not found.
- **Tree state at close**: two-tier spike reverted from `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism (as spiked)

- Long prompt (>1024 tokens): compute `gate_hit = prompt_lookup_draft(tokens, probe, 1)`, always start speculative, set `auto_probe_steps = gate_hit ? (1<<30) : 1` (or `0` in the tighter variant), print `speculation: auto gate hit/miss (...; probe skipped/N step)`.
- Decode loop: after `auto_steps > auto_probe_steps`, compute tokens/iteration and bail below 1.2. Short prompts keep the shipped 8-step probation.

## 2. Measured numbers

### Gate hits (no probe cost)

| workload | tokens | off t/s | auto t/s | ratio | gate |
|---|---:|---:|---:|---:|---|
| periodic fox x205 | 2,050 | 12.24 | 30.88 | 2.52 | hit |
| repetitive C-like, textual unique tail | 1,530 | 12.38 | 22.10 | 1.79 | hit |
| repetitive C-like, random-letter tail | 1,543 | 10.74 | 20.04 | 1.87 | hit |

### Gate-miss prose (1,643 tokens, `docs/architecture.md` 7 KB)

| probe budget | bail line | off t/s | auto t/s | ratio |
|---:|---|---:|---:|---:|
| 1 (2 iterations) | off after 2 steps (0.50/step) | 13.48 | 12.69 | 0.94 |
| 0 (1 iteration) | off after 1 steps (0.00/step) | 12.10/10.55 | 10.93/9.33 | 0.90/0.88 |

## 3. Reading

- The two-tier probe cannot be made cheap: at 2k a wasted 6-row verify is ~2-3x a plain token, so any model probe costs 5-10% on prose, while gate-off costs 0%. Bailing after one iteration is the worst case because the first verify emits nothing and is fully wasted.
- The gate itself is more robust than the earlier report assumed: "unique" tails still tokenize into repeated pieces, so every realistic repetitive-code prompt tested gate-hits. The blind spot remains theoretically possible but was not reproduced; the earlier report's 1.88x claim should be read as "forced spec wins 1.88x there", not "the gate loses 1.88x".
- **Recommendation**: keep the lookup gate as the long-prompt auto policy (0% prose cost, 1.8-2.5x on gate-hit workloads). Users with pathological late-drafting prompts can pass `--speculative` explicitly. Do not add the model probe.
- Follow-up if the blind spot ever matters: a *background* drafter-quality estimator (sample a few prompt positions offline, count 5-gram continuation matches) could predict late drafting without model work — a measurement idea for a future cycle, not a decode-path probe.

## 4. Reimplementation recipe (rebuild to re-verify the kill)

1. Apply the auto-probation patch, then the lookup gate (`reports/idea-invented-auto-spec-lookup-gate-keep.md` §7), then add `auto_probe_steps` as in §1 with `gate_hit ? (1<<30) : 1`.
2. Build + 171/171.
3. Gate-hit checks: periodic x205 and the repetitive C-like prompts (both tails) with `--speculative auto` -> expect `gate hit`, 1.8-2.5x.
4. Gate-miss check: 7 KB of `docs/architecture.md` -> expect `gate miss`, probe bail, 0.94x (budget 1) or 0.90x (budget 0).
5. Revert. Do not ship the probe; keep the gate.

## 5. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge generate ... --context 2560 --n 32 --kv f16 --threads 10 --gpu off [--speculative auto]
  periodic 2,050 -> 2.52x gate hit; code 1,530 -> 1.79x gate hit; code 1,543 -> 1.87x gate hit
  prose 1,643, probe budget 1 -> 0.94; budget 0 -> 0.90/0.88
git checkout -- apps/cli/main.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the rebuilt binary; the committed tree is unchanged except this report.
