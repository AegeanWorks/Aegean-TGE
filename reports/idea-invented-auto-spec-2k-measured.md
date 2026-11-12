# MEASURED: `--speculative auto` probation at 2k context — periodic stays on (2.4-3.1x), prose pays 5-8% while it bails; probation length is not the lever

- **Question**: The auto-speculation probation (`--speculative auto`: start speculative, bail after 8 decode iterations if tokens/iteration < 1.2) was only measured at short prompts, where prose cost 0-2%. At 2k the warm-up iterations are much more expensive per step (attention ~21% of decode), so the probation itself may cost more. Measure periodic (should stay on) and prose (should bail) at 2,050-token prompts, and test whether a shorter 4-iteration probation reduces the prose cost.
- **Cycle date**: 2026-11-11
- **Verdict**: **MEASURED — the policy works but its long-context warm-up cost is real.** Llama-3.2-3B-Instruct-Q4_K_M, `--context 2560 --n 32 --kv f16 --threads 10 --gpu off`, 2,050-token prompts:
  - **Periodic**: auto stays on (start line only, no bail), **30.25/28.61 t/s** vs off 12.62/11.77 -> **2.40/2.43x**, ids identical (`4C-1C-DE…`). Repeat window: 30.08/30.80 vs 9.67/11.21 -> 3.11/2.75x.
  - **Prose (architecture text, 7 KB)**: auto bails at **9 steps, 0.89 tokens/step**; 12.75/12.69 t/s vs off 13.10/13.48 -> **0.973/0.941**; later window 10.81/11.53 vs 12.51/12.06 -> 0.864/0.956. Median across four pairs **~0.948** (5.2% loss), ids identical (`87-63-D9…`).
  - **4-iteration probation** (spike: `auto_steps > 3`): prose bails at 4 steps (0.75/step) with ratios 0.925/0.923 in a slower window; periodic still stays on (3.11/2.75x vs off). Cross-window noise (off arm drifted 13.1 -> 11.0 t/s) prevents claiming 4 beats 9; both are in the 5-8% loss band.
  - No id changes anywhere; the bail line prints every time (`speculation: auto off after 9 steps (0.89 tokens/step)`).
- **Tree state at close**: auto-probation spike reverted from `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

Same probation recipe as `reports/idea-invented-speculation-auto-probation-keep.md`: parse `--speculative auto`, start speculative, count decode iterations while auto is live, compute `tokens/iteration` after the probation window, and disable both spec flags once `per_step < 1.2`. The probe counts *loop iterations* (not spec steps), so it measures the draft source's real yield. At 2k each iteration is ~2-3x slower than at short context because the 6-row verify reads a 2k KV cache, so the cost of the probation window scales with prompt length while the benefit does not (prose has no drafts).

## 2. Results

### Periodic 2,050-token prompt

| pair | off t/s | auto t/s | ratio | bail | ids |
|---:|---:|---:|---:|---|---|
| 1 | 12.62 | 30.25 | 2.40 | none | identical |
| 2 | 11.77 | 28.61 | 2.43 | none | identical |
| 3 | 9.67 | 30.08 | 3.11 | none | - |
| 4 | 11.21 | 30.80 | 2.75 | none | - |

### Prose 2,050-token prompt (`docs/architecture.md` first 7 KB)

| probation | pair | off t/s | auto t/s | ratio | bail |
|---|---:|---:|---:|---:|---|
| 8 iterations | 1 | 13.10 | 12.75 | 0.973 | 9 steps, 0.89/step |
| 8 iterations | 2 | 13.48 | 12.69 | 0.941 | 9 steps, 0.89/step |
| 8 iterations | 3 | 12.51 | 10.81 | 0.864 | 9 steps, 0.89/step |
| 8 iterations | 4 | 12.06 | 11.53 | 0.956 | 9 steps, 0.89/step |
| **8 iter median** | | | | **0.948** | |
| 4 iterations | 1 | 11.27 | 10.43 | 0.925 | 4 steps, 0.75/step |
| 4 iterations | 2 | 10.96 | 10.12 | 0.923 | 4 steps, 0.75/step |

## 3. Reading

- Auto-spec does its job at 2k: repetitive prompts stay speculative (2.4-3.1x), free prose bails and the output is byte-identical (`87-63-D9`).
- The prose cost is ~5-8% at 2k, versus 0-2% at short prompts: the 9 warm-up iterations at ~10-13 t/s cost ~0.7-0.9 s of the 2.5-2.9 s run. The 4-iteration variant burns fewer iterations but each probe step is slower and the measured ratios do not improve beyond noise; probation length is not the effective lever.
- An effective lever would be **prompt-length gating**: skip auto entirely above ~1k tokens (leave spec off unless the user forces it), or probe with 2 single-row steps (no verify batch) — the latter is a new mechanism, not a constant change. Either keeps the 2.4x repetitive win (users can still pass `--speculative`) and removes the 5-8% prose tax at long context.
- Usage: at 2k, prefer `--speculative` explicitly for repetitive/templated prompts and leave auto for ≤1k prompts; the current auto policy is net-positive on average only if most long prompts are repetitive.

## 4. Reimplementation recipe

1. Re-apply the auto-probation patch (`reports/idea-invented-speculation-auto-probation-keep.md` §8: tri-state flag, mutable spec flags, iteration counter, 8-step bail, attention pin).
2. Build + 171/171.
3. Periodic: `generate --prompt "<fox x205>" --context 2560 --n 32 --kv f16 --threads 10 --gpu off --speculative auto` vs off; expect 2.4-3.1x, no bail line, ids identical.
4. Prose: same with the first 7 KB of `docs/architecture.md`; expect `speculation: auto off after 9 steps (0.89 tokens/step)` and 0.94-0.97x.
5. Variant: change the bail threshold to `auto_steps > 3`; expect 4-step bail (0.75/step), 0.92-0.93x in the same window — no reliable improvement.
6. Revert. The recommendation (prompt-length gate or 2-step probe) is a follow-up idea, not this cycle.

## 5. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge generate ... --context 2560 --n 32 --kv f16 --threads 10 --gpu off [--speculative auto]
  periodic -> 2.40/2.43x (later window 3.11/2.75x), no bail, ids identical
  prose    -> 0.973/0.941/0.864/0.956, bail 9 steps 0.89/step, ids identical
  4-step variant -> 0.925/0.923, bail 4 steps 0.75/step
git checkout -- apps/cli/main.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the rebuilt binary; the committed tree is unchanged except this report.
