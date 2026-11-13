# MEASURED: 12k speculation is workload-dependent — code still wins 1.31-1.35x at 3.50 accepted/step, so the lock-prompt 0.84x is an acceptance artifact

- **Idea**: `idea-invented-spec-12k-killed.md` measured 0.84x at 12k on the only fit-capable periodic prompt (a period-1 " a" lock) and concluded speculation loses at 12k. That prompt's acceptance collapsed to 1.50/step; the report itself flagged the verdict as workload-specific. The fair test is the code workload, which is token-dense enough to reach 12k within the Windows command line (2.14 chars/token) and accepted 3.00/step at 8k. Measure off vs `--speculative` at 12,284 tokens of repetitive C-like code.
- **Cycle date**: 2026-11-13
- **Verdict**: **MEASURED — speculation wins at 12k code: 5.24 vs 3.87-4.00 t/s (1.31-1.35x).** Llama-3.2-3B-Instruct-Q4_K_M, 12,284-token code prompt (323 C functions + unique tail), `--context 12800 --n 16 --kv f16 --threads 10 --gpu off`:
  - **off**: **4.00 t/s** (prefill 407.8 s) and **3.87 t/s** (prefill 474.3 s, later window); decode attn 2498-2558 ms (66-67%).
  - **`--speculative`**: **5.24 t/s**; `2 steps, 7 draft tokens accepted (3.50/step), 9 emitted`; decode attn 1860 ms (65%). Ratio **1.31x** against the faster off, **1.35x** against the slower off; interpolated at the spec arm's window (~3.93 t/s) it is **~1.33x**.
  - **The decode comparison is per-row matched despite prefill drift**: decode matvec per row 79.4 ms (spec) vs 80.2 / 84.7 ms (off) and attention per row 155 ms vs 166.5 / 170.5 ms. The prefill totals drifted 408 -> 444 -> 474 s across the three runs, but the decode phases ran at comparable per-row cost, so the 1.3x is not a drift artifact.
  - **The 12k lock result stands as measured but is not a context verdict**: same context, same flags, 12k lock accepted 1.50/step and lost 0.84x; 12k code accepts 3.50/step and wins 1.33x. Speculation's viability at 12k is set by accepted tokens per verify, not by context length.
- **Tree state at close**: no code touched (shipped flags); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Llama-3.2-3B-Instruct-Q4_K_M.gguf |
| prompt | 323 x `int f$i(int x) { int y = x * $i + 7; if (y > 100) { y = y - 100; } return y; }` + `// unique tail 8f3a2c` = **12,284 tokens**, 26,291 chars |
| context / n / KV / threads / GPU | 12800 / 16 / f16 / 10 / off |
| arms | off (2 runs), `--speculative` (1 run) |

The prompt shape matches the 8k code workload (`idea-invented-code-8k-spec-sparse-measured.md`: 8,104 tokens, 3.00 accepted/step, 1.13-1.14x), scaled to the 12k cell. The code family is the only one that fits 12k in the one-argument command line: natural prose is ~3.6 chars/token (12k needs ~43k chars) and the periodic fox text ~3.5; this prompt is 2.14 chars/token.

## 2. Results

| run | arm | prefill | decode t/s | decode attn | spec line | matvec / attn per row |
|---|---|---|---:|---|---|---|
| 1 | off | 407.8 s | 4.00 | 2498 ms (67%) | — | 80.2 / 166.5 ms |
| 2 | spec | 444.5 s | **5.24** | 1860 ms (65%) | 2 steps, 7 acc (3.50/step), 9 emitted | 79.4 / 155.0 ms |
| 3 | off | 474.3 s | 3.87 | 2558 ms (66%) | — | 84.7 / 170.5 ms |

Ratios: spec/off = **1.31x** (run 1), **1.35x** (run 3); at the spec run's window (interpolated off ~3.93) **~1.33x**. The spec line is deterministic (2 steps, 7 accepted), and its per-row costs agree with both off runs, so the ratio is admissible despite the prefill drift.

### 12k spec across workloads

| workload | accepted/step | spec vs off |
|---|---:|---:|
| lock-1 (`a` + ` a`x12,288) | 1.50 | **0.84x** (loss) |
| code (323 functions) | 3.50 | **1.31-1.35x** (win) |
| 8k code (reference) | 3.00 | 1.13x |

The two 12k points bracket the boundary: the verify's six rows at 12k cost more than at 8k (attention 65-67% of decode), so the break-even acceptance is higher than 8k's. At 1.50/step the verify loses; at 3.50/step it wins comfortably.

## 3. Reading

- **Do not read the 12k lock result as "speculation is dead past 8k"**. It is one workload whose model continuation stopped repeating; the verifier correctly rejected the drafter. The code workload keeps drafting at 12k and wins 1.33x.
- **The 12k boundary is acceptance-driven**: from the measured points, 12k break-even sits between 1.50 and 3.50 accepted/step (8k break-even is below 3.00). Quote acceptance with any context claim.
- **Prefill at 12k remains the wall**: 408-474 s at 26-30 t/s, attention 44-46% of it. Speculation does not touch prefill; the 1.33x decode win is invisible in a 16-token run's wall, where prefill is >99%. For long generations the decode win matters; for short outputs the prefill does.
- **Guidance**: at 12k, enable `--speculative` for code/template/echo workloads after a sample check of `spec:` accepted/step (>= ~3 wins); for lock-like or novel text at 12k, leave it off or accept a possible loss. The 8k matrix's recommendation is unchanged at 8k.
- **Method note**: the only 12k prompts available are code and lock (command-line limit); prose and fox cannot reach 12k without a prompt-file flag. The two available families disagree, which is why the report quotes both rather than a single 12k verdict.

## 4. Falsifier (stated before the runs)

**CONFIRM the lock result is workload-specific** if spec >= 1.10x off on 12k code. **GENERALIZE the 12k kill** if spec is <= 1.0x on 12k code too. Measured 1.31-1.35x: confirmed workload-specific. The 12k lock report's measured ratio stands; its implication for other 12k workloads does not.

## 5. File anchors (current HEAD, read-before-write verified)

- `apps/cli/main.cpp:788-922` — fused greedy chain measured (six-row verify).
- `include/tge/models/qwen2.hpp:180` — `kSpeculativeBatch = 6`.
- `src/cpu/avx2/attention_avx2.cpp:660-864` — block kernel; the six-row KV sharing sets the 12k verify cost.
- `reports/idea-invented-spec-12k-killed.md` — the 12k lock point this cycle contextualizes.
- `reports/idea-invented-code-8k-spec-sparse-measured.md` — the 8k code row.
- `reports/idea-invented-spec-at-8k-measured.md`, `idea-invented-spec-prose-8k-kills-lookup-gate.md` — the 8k periodic/prose rows.

## 6. Reimplementation recipe

1. Build the prompt (26,291 chars; fits the one-argument limit):
   ```powershell
   $sb = New-Object System.Text.StringBuilder
   foreach ($i in 1..323) {
     [void]$sb.AppendLine("int f$i(int x) { int y = x * $i + 7; if (y > 100) { y = y - 100; } return y; }")
   }
   [void]$sb.Append("// unique tail 8f3a2c")
   $p = $sb.ToString()   # 12,284 tokens
   ```
2. Run the arms one per shell call (each ~7-8 min of prefill; two runs exceed the 15-minute cap):
   ```powershell
   tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt $p \
     --context 12800 --n 16 --kv f16 --threads 10 --gpu off [--speculative]
   ```
   Expect off 3.87-4.00 t/s and spec 5.24 t/s with `2 steps, 7 accepted (3.50/step), 9 emitted`; decode attn 65-67%.
3. Compare per-row phases if prefill drifts: decode matvec/row ~79-85 ms and attn/row ~155-171 ms should match across arms.
4. No code ships.

## 7. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56, cycle gate)
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed

12k code (ctx 12800, n 16, f16, 10 threads, gpu off):
  off  : prefill 407.8 s ; decode 4.00 t/s (attn 2498 ms, 67% ; 80.2/166.5 ms per row)
  spec : prefill 444.5 s ; decode 5.24 t/s (attn 1860 ms, 65% ; 79.4/155.0 ms per row)
         2 steps, 7 draft tokens accepted (3.50/step), 9 tokens emitted
  off  : prefill 474.3 s ; decode 3.87 t/s (attn 2558 ms, 66% ; 84.7/170.5 ms per row)
  ratios spec/off: 1.31x (run 1), 1.35x (run 3); ~1.33x interpolated at run 2's window
git status --short                                -> clean (no code touched)
python aikit.py check aegean-tge                  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
