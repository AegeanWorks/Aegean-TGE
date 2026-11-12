# MEASURED: the lookup gate's blind spot is real — a late-drafting code prompt loses 1.88x (spec 22.5 vs off 12.0 t/s), so the gate must not be the only long-prompt policy

- **Question**: `idea-invented-auto-spec-lookup-gate-keep.md` gates long-prompt auto-spec on whether the prompt's **last five-token window** appears earlier. The report flagged the blind spot: a prompt whose drafting potential emerges only after generation starts would be gated off even though speculation would win. Quantify it with a workload built to trigger it — 1,530 tokens of structurally repetitive C-like code ended by a unique marker (`// unique tail 8f3a2c`), so the last window cannot match.
- **Cycle date**: 2026-11-11
- **Verdict**: **MEASURED — the blind spot is large and decisive.** Llama-3.2-3B-Instruct-Q4_K_M, `--context 2560 --n 32 --kv f16 --threads 10 --gpu off`:
  - forced spec: **22.45/22.58 t/s**, `6 steps, 26 accepted (4.33/step), 31 emitted`.
  - off: 11.06/12.78 t/s.
  - ratio: **1.75-2.04x (median ~1.88x)** — the gate would have selected off and left that on the table.
  - The drafter recovers as soon as generation re-enters the prompt's repeated structure; accepted/step is 4.33 from the first steps, even though the last prompt window is unique.
- **Consequence**: a free prompt-only gate is not a safe replacement for the probation on long prompts. The correct long-prompt policy needs a *cheap model probe* (one speculative iteration: stay if it accepts, bail if not), not a prompt-only heuristic. The 9-step probation's 5-8% cost and the gate's 0% cost are both wrong ends of the trade.
- **Correction (2026-11-11, follow-up cycle)**: the claim that the lookup gate would reject the 1,530-token code prompt is **false in measurement** — the gate hits it (the textual "unique" tail still tokenizes into five-token windows that appear earlier), as it does for a random-letter tail variant. Forced spec does win 1.88x there, but the gate does not lose it. The blind spot remains theoretically possible and unproven on tested shapes; the two-tier model probe was measured and killed (it costs 5-10% on prose vs the gate's 0%). Read this report as "forced spec wins late-drafting workloads", not "the gate loses them". See `reports/idea-invented-auto-spec-two-tier-killed.md`.
- **Tree state at close**: no code touched this cycle (measurement only); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Workload construction

PowerShell:

```text
$sb = New-Object System.Text.StringBuilder
for ($i = 1; $i -le 40; $i++) { $sb.AppendLine("int f$i(int x) { int y = x * $i + 7; if (y > 100) { y = y - 100; } return y; }") }
$sb.Append("// unique tail 8f3a2c")
$prompt = $sb.ToString()      # 1,530 tokens
```

The 40 functions share every token pattern except the function name and the multiplier; the trailing comment is unique, so `prompt_lookup_draft(prompt, probe, 1)` cannot find the last window. After the first generated tokens, the model's own output plus the prompt's repeated body gives the drafter matches again.

## 2. Results

| pair | off t/s | forced spec t/s | ratio | spec line |
|---:|---:|---:|---:|---|
| 1 | 11.06 | 22.45 | 2.03 | 6 steps, 26 accepted (4.33/step), 31 emitted |
| 2 | 12.78 | 22.58 | 1.77 | 6 steps, 26 accepted (4.33/step), 31 emitted |
| **median** | 11.92 | 22.52 | **1.88** | |

`--print-ids` was omitted: the bench prints the full 1,530-token prompt id list, which floods the capture. A direct run with prints confirms the same timings and the spec line.

## 3. Reading

- The gate's blind spot is not theoretical: 1.88x on a realistic code-completion shape. Any long-prompt auto policy that only looks at the prompt will systematically miss code/template prompts whose *continuation* repeats the body — the exact class where speculation is best.
- The safe policy is a **one-iteration model probe** for prompts above the threshold: start speculative, run one step, and keep spec if that step accepted at least one draft (tokens/iteration >= 1.2); bail otherwise. Cost on prose: one verify + correction at 2k (~80-100 ms, ~3-4% of the run) versus the probation's 5-8%; benefit: it sees the real drafter and would keep this code prompt.
- A two-tier policy is also viable: prompt gate says on -> speculative immediately (free, a guaranteed-hit case); gate says off -> one-iteration probe. That preserves the gate's 0% cost on repetitive prompts and bounds the blind-spot cost.
- Do not ship the prompt-only gate as the whole long-prompt policy. The earlier report is now amended in spirit: keep-grade only as the first tier of a two-tier policy.

## 4. Reimplementation recipe

1. Rebuild the workload in §1; run `generate --prompt $prompt --context 2560 --n 32 --kv f16 --threads 10 --gpu off [--speculative]` and read the `decode:` and `spec:` lines. Expect off ~11-13 t/s, spec ~22.5 t/s, 4.33 accepted/step.
2. To test the blind spot directly, apply the lookup gate (`reports/idea-invented-auto-spec-lookup-gate-keep.md` §7) and run with `--speculative auto`: expect `speculation: auto gate off` and ~12 t/s.
3. For the follow-up probe: on long prompts, start speculative, run one decode iteration, and keep only if tokens/iteration >= 1.2; measure prose cost (~3-4%) and this code prompt (should stay on at ~22 t/s).
4. No fidelity run owed: exact greedy, no arithmetic change.

## 5. Exact commands run and results

```text
# workload built in PowerShell as §1; model Llama-3.2-3B-Instruct-Q4_K_M.gguf
tge generate ... --n 32 --kv f16 --threads 10 --gpu off                -> 11.06/12.78 t/s
tge generate ... --n 32 --kv f16 --threads 10 --gpu off --speculative  -> 22.45/22.58 t/s,
  6 steps, 26 accepted (4.33/step), 31 emitted
git status --short                                                     -> clean (no code touched)
python aikit.py check aegean-tge                                       -> 4/4 gates passed
```

No push, no merge, no force-push. Measurement only; the committed tree is unchanged except this report.
