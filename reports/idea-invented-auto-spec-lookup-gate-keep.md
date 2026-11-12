# INVENTED: long-prompt lookup gate for `--speculative auto` — KEEP-GRADE (2k periodic 2.87x, 2k prose 1.06x with zero spec cost, short prompts unchanged)

- **Idea**: The auto-probation report measured a 5-8% tax on long prose because the 9-step probation runs expensive 2k-context verify iterations before bailing. Replace the probation for long prompts with a zero-model gate: before the loop, test whether the prompt's **last five tokens appeared earlier** (`prompt_lookup_draft(tokens, probe, 1)`), which is exactly the condition the drafter needs at the first position. Gate on -> speculative; gate off -> plain decode from token 0. Short prompts (<=1024 tokens) keep the probation, which is cheap there and catches drafting that only starts later in the stream.
- **Cycle date**: 2026-11-11
- **Verdict**: **KEEP-GRADE (report-only; promotion human-gated).** Llama-3.2-3B-Instruct-Q4_K_M, `--context 2560 --n 32 --kv f16 --threads 10 --gpu off`:
  - **2,050-token periodic**: gate **on** -> 28.41/30.06 t/s vs off 9.84/10.51 -> **2.89/2.86x**, no probation, ids identical (`4C-1C-DE…`).
  - **1,643-token prose** (`docs/architecture.md` 7 KB): gate **off** -> 10.95/13.20 vs 10.33/12.26 -> 1.06/1.08x (plain decode; the >1 ratios are within-pair drift), ids identical (`87-63-D9…`). The 5-8% probation tax is gone.
  - **24-token prose**: unchanged probation path -> `speculation: auto off after 9 steps (0.89 tokens/step)`; ratios 1.04/0.94 in this window, ids identical (`12-62-94…`).
- **Tree state at close**: gate spike reverted from `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

- `run_generate` parses `--speculative auto` (tri-state) and passes `speculative_auto` to `command_generate`.
- After prompt tokenisation, before the decode loop:
  - `tokens.size() > 1024`: `int probe[1]; speculative = prompt_lookup_draft(tokens, probe, 1);` and print `speculation: auto gate on/off (last-window lookup, N prompt tokens)`.
  - otherwise: the shipped probation (start speculative, 8 iterations, bail below 1.2 tokens/step).
- Attention pins to Head only when the gate/probation leaves speculation on.
- The lookup is O(prompt) string-free comparison over the existing drafter and runs before any model work, so the gate costs microseconds.

## 2. Predicted gain

At 2k the probation cost ~0.7-0.9 s on prose (5-8%). The gate removes that entirely when the last window does not match, and keeps the 2.4-3x when it does. Prediction: prose 2k -> ~1.00x vs off, periodic 2k -> >=2.4x, short prompts unchanged.

Measured exactly that (prose 1.06/1.08 with drift; periodic 2.87x; short unchanged).

## 3. Falsifier

**KEEP** if (a) 2k periodic auto >= 2x off with ids identical, (b) 2k prose auto within 2% of off (or better) with ids identical, and (c) short-prompt auto keeps the probation and its ~1.0x behaviour. **KILL** if the gate hides a drafting case that the probation would have caught on a workload in the fixed set, or if any ids change.

All three pass. Known limitation, same as the earlier static-probe attempt: a long prompt whose *later* generated text would repeat even though the prompt's last window does not match will be gated off. That is the accepted trade of a zero-model gate; the user can still pass `--speculative` explicitly.

## 4. File anchors (current HEAD, read-before-write verified)

- `apps/cli/main.cpp:2160-2180` — `--speculative` parser (tri-state in the spike).
- `:690-700` — auto decision site (`const std::vector<int>& prompt = tokens;` follows it).
- `:785-800` — decode-loop head and the probation counters (spike-only).
- `prompt_lookup_draft` — the drafter used for the lookup (defined in `main.cpp`).
- `reports/idea-invented-speculation-auto-probation-keep.md` — the shipped probation.
- `reports/idea-invented-auto-spec-2k-measured.md` — the 2k cost that motivated the gate.

## 5. Measured numbers

### 2,050-token periodic (auto gate on)

| pair | off t/s | auto t/s | ratio | gate line | ids |
|---:|---:|---:|---:|---|---|
| 1 | 9.84 | 28.41 | 2.89 | gate on (2050 tokens) | identical |
| 2 | 10.51 | 30.06 | 2.86 | gate on (2050 tokens) | identical |

### 1,643-token prose (auto gate off)

| pair | off t/s | auto t/s | ratio | gate line | ids |
|---:|---:|---:|---:|---|---|
| 1 | 10.33 | 10.95 | 1.06 | gate off (1643 tokens) | identical |
| 2 | 12.26 | 13.20 | 1.08 | gate off (1643 tokens) | identical |

### 24-token prose (probation path unchanged)

| pair | off t/s | auto t/s | ratio | bail | ids |
|---:|---:|---:|---:|---|---|
| 1 | 15.07 | 15.70 | 1.04 | 9 steps, 0.89/step | identical |
| 2 | 13.23 | 12.50 | 0.94 | 9 steps, 0.89/step | identical |

## 6. Reading

- The gate is strictly better than the probation for long prompts: it removes the warm-up tax and preserves the repetitive win. The only regression surface is a long prompt whose drafting potential appears later; the probation could catch that, the gate cannot. Both probes were measured and the trade is explicit.
- The threshold (1024 tokens) is a judgment call: below it a probation step costs ~15-30 ms; above it ~70-100 ms. 1024 sits where the probation's total cost crosses ~2%.
- Promotion is human-gated; if shipped, `--speculative auto` should document the gate and the trade, and the solo-vs-busy style audit does not apply (spec is a single-sequence path).

## 7. Reimplementation recipe

1. Apply the auto-probation patch (`reports/idea-invented-speculation-auto-probation-keep.md` §8): tri-state flag, mutable spec flags, iteration counter, 8-step bail, attention pin. Keep the `speculative_auto` parameter in `command_generate`.
2. Replace the unconditional start block with the gated version in §1: if `tokens.size() > 1024`, `speculative = prompt_lookup_draft(tokens, probe, 1)` and print `speculation: auto gate ...`; else start speculative and print the probation line. Pin attention only when `speculative` is true after this block.
3. Build + 171/171.
4. Periodic 2k: `generate --prompt "<fox x205>" --context 2560 --n 32 --kv f16 --threads 10 --gpu off --speculative auto` -> expect gate on, 2.4-2.9x, ids identical.
5. Prose 1.6k: same with `docs/architecture.md` -> expect gate off, ~1.0x, ids identical.
6. Short prose: same with a 24-token prompt -> expect the probation bail line, ~1.0x.
7. Revert or ship after the human gate; document the gate/trade.

## 8. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge generate ... --context 2560 --n 32 --kv f16 --threads 10 --gpu off [--speculative auto]
  periodic 2k -> 2.89/2.86x, gate on, ids identical
  prose 1.6k  -> 1.06/1.08x, gate off, ids identical
  prose 24    -> probation bail 9 steps, 1.04/0.94, ids identical
git checkout -- apps/cli/main.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the rebuilt binary; the committed tree is unchanged except this report.
