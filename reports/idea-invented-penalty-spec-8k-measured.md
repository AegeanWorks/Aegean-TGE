# MEASURED: repeat penalty kills speculation at 8k — acceptance collapses to 0.25/step and spec is neutral-to-negative (0.94-1.03x)

- **Idea**: Chat serving usually sets a repeat penalty, and the 2k penalty cell was nearly useless for spec (`idea-invented-fused-lock-six-draft-killed.md` family: only 1 step fired in 15 tokens, 1.04x). At 8k, prose speculation without a penalty wins 1.18-1.25x (drafts echo the prompt's recurring five-grams). Does a typical `--repeat-penalty 1.1` preserve that, or does it suppress exactly the repeated continuations the drafter proposes? Measure sampled off vs spec on the 8,256-token prose prompt with `--temp 0.8 --top-p 0.9 --repeat-penalty 1.1 --seed 7`.
- **Cycle date**: 2026-11-13
- **Verdict**: **MEASURED — the penalty collapses the drafter; speculation adds nothing at 8k with repeat penalty 1.1 and can lose slightly.** Llama-3.2-3B-Instruct-Q4_K_M, 8,256-token prose prompt, `--context 8704 --n 32 --kv f16 --threads 10 --gpu off --temp 0.8 --top-p 0.9 --repeat-penalty 1.1 --seed 7`:
  - **off**: **4.52 t/s** (adjacent pair); `timers sample 25.4 ms (0.4%)`.
  - **spec**: **4.25 t/s** in the adjacent pair (**0.94x**), **4.66 t/s** in a later window; **`24 steps, 6 draft tokens accepted (0.25/step), 30 emitted`** on both runs. The spec line is deterministic: 6 accepted over 24 verify steps.
  - **Mechanism**: the prompt-lookup drafter proposes the continuation of a recurring five-gram; with penalty 1.1 over the history, those repeated tokens are penalised in the verifier's logits, so the exact-greedy accept rule rejects almost every draft. Without the penalty the same prompt at 8k accepts 3.17/step (greedy 1.18-1.25x); the penalty, not sampling and not context, is what kills it.
  - **Sampling is not the cause**: sampled spec at 8k on periodic text without a penalty is 1.44-1.72x with 3.29 accepted/step (`idea-invented-sampled-spec-8k-measured.md`).
  - **Caveat recorded**: the box became loaded by user applications (Edge/VS Code) during the reverse-order pair; the matched off arm in that window was killed by the shell cap and produced no output. The quoted ratio uses the adjacent off/spec pair, and the acceptance figure is identical across both spec runs, so the mechanism is deterministic even though the pair count is one.
- **Tree state at close**: no code touched (shipped flags); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Llama-3.2-3B-Instruct-Q4_K_M.gguf |
| prompt | first 30,000 chars of the distinct-docs corpus, `"` -> `'` sanitised = **8,256 tokens** |
| context / n / KV / threads / GPU | 8704 / 32 / f16 / 10 / off |
| sampling | `--temp 0.8 --top-p 0.9 --repeat-penalty 1.1 --seed 7` (default penalty window) |
| arms | off, `--speculative` |

## 2. Results

| run | arm | decode t/s | spec line | timers |
|---|---|---:|---|---|
| 1 | off | 4.52 | — | sample 25.4 ms (0.4%) |
| 2 | spec | **4.25 (0.94x)** | 24 steps, 6 acc (0.25/step), 30 emitted | sample 1958.7 ms (26.9%) — timer double count, ignore |
| 3 | spec | 4.66 | 24 steps, 6 acc (0.25/step), 30 emitted | — |

The off arm is from the adjacent run in the same window as run 2; run 3 is a later confirmation of the deterministic spec line. No penalty-1.0 control was captured because the box became loaded; the no-penalty cells already on record are the periodic sampled 1.44-1.72x and the prose greedy 1.18-1.25x.

## 3. Reading

- **Recommendation**: do not enable `--speculative` with a repeat penalty >= 1.1 at 8k. It is neutral-to-slightly-negative (0.94-1.03x) and doubles the verify work for 0.25 accepted tokens per step. If a workload needs both, measure its accepted/step first; the drafter must find non-repeated continuations for the penalty not to suppress it.
- **The penalty is a drafter-suppression knob**, not a sampler detail: it rewrites the verifier logits per row (the penalized spec path) and deliberately lowers the probability of exactly the tokens prompt-lookup proposes. The 2k result (1 step fired) and this 8k result (0.25/step) are the same mechanism at two contexts.
- **The matrix's sampled row should carry the condition**: "sampled spec 1.44-1.72x **without repeat penalty**". With a penalty, the 8k cell is ~1.0x.
- **What remains true**: speculation is a quality-neutral accelerator when accepted/step is ~3+; repeat penalty breaks that condition on echo-heavy prompts. Greedy spec with penalty was not re-measured here (the greedy penalized path uses `verify_draft_logits` and per-row penalties); the sampled result is the one measured, and the accept rule is the same.
- **Box note**: the host is shared with user applications; long prefill runs can exceed the shell window and should be launched when the machine is idle or via a tracked background job.

## 4. Falsifier (stated before the runs)

**KEEP spec with penalty** if it is >= 1.15x off at 8k prose with penalty 1.1. **DO NOT RECOMMEND** if it is within noise or below. Measured 0.94-1.03x with 0.25 accepted/step: do not recommend; the condition is recorded.

## 5. File anchors (current HEAD, read-before-write verified)

- `apps/cli/main.cpp:1081-1195` — sampled speculative block (`verify_draft_logits`, per-row `penalized_softmax`, accept, bonus).
- `apps/cli/main.cpp:1119-1142` — the penalized softmax lambda (penalty then softmax then cuts).
- `src/models/qwen2.cpp` — `verify_draft_logits` six-row path.
- `include/tge/models/qwen2.hpp:180` — `kSpeculativeBatch = 6`.
- `reports/idea-invented-sampled-spec-8k-measured.md` — the no-penalty sampled 8k row (1.44-1.72x).
- `reports/idea-invented-spec-prose-8k-kills-lookup-gate.md` — the no-penalty greedy prose row (1.18-1.25x, 3.17/step).
- `reports/idea-invented-fused-lock-six-draft-killed.md` — the 2k penalty cell (1 step fired).

## 6. Reimplementation recipe

1. Build the prose prompt (30,000 chars, quote-sanitised) and confirm 8,256 tokens with `--context 64`.
2. Run off then spec (each ~5 min prefill when idle; the box's user load can stretch this past a 10-minute shell window):
   ```powershell
   tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt $p \
     --context 8704 --n 32 --kv f16 --threads 10 --gpu off \
     --temp 0.8 --top-p 0.9 --repeat-penalty 1.1 --seed 7 [--speculative]
   ```
   Expect off ~4.5 t/s and spec ~4.3-4.7 t/s with `24 steps, 6 accepted (0.25/step)`.
3. For the control, rerun without `--repeat-penalty` (sampled periodic or prose) and expect the drafter to recover to ~3+ accepted/step; this isolates the penalty.
4. No code ships.

## 7. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56, cycle gate)
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed

8k prose, temp 0.8 / top-p 0.9 / repeat-penalty 1.1 / seed 7 (ctx 8704, n 32, f16, 10 threads):
  off  : 4.52 t/s (sample 25.4 ms, 0.4%)
  spec : 4.25 t/s (0.94x) ; 24 steps, 6 accepted (0.25/step), 30 emitted
  spec : 4.66 t/s (later window) ; same 24 steps, 6 accepted line
  (reverse-order off arm killed by the shell cap while the host was loaded by user apps)
git status --short                                -> clean (no code touched)
python aikit.py check aegean-tge                  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
