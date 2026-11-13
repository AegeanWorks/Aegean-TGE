# MEASURED + KILLED: forced speculation wins on 8k prose (1.18-1.25x), so the zero-model lookup gate is wrong for long prompts

- **Idea**: The auto-speculation lookup gate (`idea-invented-auto-spec-lookup-gate-keep.md`) was keep-grade at 2k: for long prompts it probes `prompt_lookup_draft(tokens, probe, 1)` on the prompt's last five tokens, runs speculative only if that window appeared earlier, and otherwise decodes plainly — removing the 5-8% probation tax on prose. It was never shipped. The open question at 8k: does the gate's premise hold when the prompt is long? If forced speculation on 8k prose is a loss, the gate is the fix; if it is a win even when the last window does not match, the gate would suppress real speed. Measure forced spec vs off on an 8,256-token non-repeating prose prompt first.
- **Cycle date**: 2026-11-13
- **Verdict**: **MEASURED — forced speculation is 1.18-1.25x on 8k prose with 3.17 accepted/step; the lookup gate is KILLED for long prompts.** Llama-3.2-3B-Instruct-Q4_K_M, 8,256-token prose prompt (eight distinct docs, 30,000 chars), `--context 8704 --n 32 --kv f16 --threads 10 --gpu off`, interleaved:
  - **off**: **5.78 / 5.86 t/s**.
  - **`--speculative`**: **7.25 / 6.91 t/s** -> **1.25x / 1.18x** (median ~1.22x); `6 steps, 19 draft tokens accepted (3.17/step), 24 emitted`, identical on both runs.
  - **The gate would have turned this off**: it decides from the prompt's last five tokens only, and non-repeating prose's tail need not match earlier text. At 8k the drafter still yields 3.17 tokens/step because the 8k history contains many recurring five-grams; the model quickly falls into phrasing it used earlier in the same prompt, and each such recurrence is draftable. A zero-model t=0 probe cannot see that.
  - **Why 2k disagreed**: the 2k gate measurement used 1,643-token prose, where the drafter died after a few steps (forced spec measured ~neutral there, 0.985x in the later 2k prose check). At 8k the same drafter keeps finding matches for the whole 32-token run. The gate's keep verdict was correct for its 2k corpus and does not transfer to long prompts.
  - **No code change**: the auto mode was never shipped (`--speculative` is a boolean at `apps/cli/main.cpp:2165-2167`), so nothing to revert. The recommendation is explicit: do not promote the lookup gate for long prompts; keep `--speculative` on by default for 8k use.
- **Tree state at close**: no code touched (measurement only); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Llama-3.2-3B-Instruct-Q4_K_M.gguf |
| prompt | first 30,000 chars of the distinct-docs corpus (`architecture.md`, `report-0.1.md`, `report-0.2-verification.md`, `benchmarks.md`, `roadmap.md`, `hardware-support.md`, `fused-correction.md`, `README.md`), double quotes replaced with single quotes so the one-argument command line stays intact |
| prompt size | **8,256 tokens** (counted with `--context 64`) |
| context / n / KV / threads / GPU | 8704 / 32 / f16 / 10 / off |
| arms | off, `--speculative` (interleaved: off->spec, spec->off) |

The prompt is natural prose with no document repetition; the only edit is quote sanitization (the CLI has no prompt-file input, and embedded `"` breaks the quoted CreateProcess argument).

## 2. Results

| pair | arm | decode t/s | ratio | spec line |
|---|---|---:|---:|---|
| 1 | off | 5.78 | — | — |
| 1 | spec | **7.25** | **1.25x** | 6 steps, 19 acc (3.17/step), 24 emitted |
| 2 | spec | **6.91** | **1.18x** | 6 steps, 19 acc (3.17/step), 24 emitted |
| 2 | off | 5.86 | — | — |

Median of the four cross-pairings: **~1.22x**. Both spec runs are identical in the spec line, so the drafter's yield is deterministic on this prompt; the off arm is stable (5.78/5.86), so the pairs are admissible.

### Why the drafter works at 8k but not at 2k

| workload | history for the drafter | measured yield | forced-spec result |
|---|---:|---|---|
| 1,643-token prose (gate cycle) | ~1.6k | drafts die after ~3 steps | ~neutral (0.985x in the later check) |
| 8,256-token prose (this cycle) | ~8.3k | 3.17 accepted/step over 6 steps | **1.18-1.25x** |
| 8,104-token code | ~8.1k | 3.00 accepted/step over 8 steps | 1.13-1.14x |
| 8,193-token periodic | ~8.2k | 3.00 accepted/step over 4-8 steps | 1.42-1.61x |

The mechanism is history size: a prompt-lookup drafter finds a five-token window by exact match, so a longer prompt contains more recurring phrasing and more chances for the generated text to match an earlier occurrence. The model's own repeated phrasing (not just literal document repetition) is the draft source. That is precisely what a t=0 last-window probe cannot measure.

## 3. Reading

- **`--speculative` is an 8k win on every measured workload**: periodic 1.42-1.61x, prose 1.18-1.25x, code 1.13x. It is quality-neutral for greedy (verifier-exact) and works without a gate. The 8k decision matrix should list it as the default accelerator, not a conditional one.
- **The lookup gate is context-bounded**: it was a correct fix for 2k prose's probation tax, where the drafter genuinely dies. At 8k the same prose keeps drafting, so gating on the last prompt window would trade a ~1.2x win for a few hundred milliseconds of probe savings. Do not promote it as a long-prompt policy.
- **The auto-probation idea inherits the same problem**: any t=0 bail decision underestimates a drafter whose yield grows with history. If an auto mode is ever revisited, the gate must sample the generated stream (or run the probation at all lengths) rather than the prompt tail.
- **The prose number also bounds the "spec is only for repetitive text" intuition**: at 8k, natural docs are draftable. The accelerating stack remains f16 KV + `--speculative`; add `--sparse` only with its 13-19pp natural-text quality cost.
- **Method note**: the 30,000-char prompt is near the Windows command-line limit (30,144 chars total). Any longer prose prompt requires quote sanitization and careful length budgeting, or a future prompt-file flag.

## 4. Falsifier

**KILL the lookup gate for long prompts** if forced speculation on 8k prose is >= 1.10x off while the prompt's last window is not guaranteed to match. **KEEP the gate** if forced spec is within noise of off (then its t=0 decision costs nothing). Measured 1.18-1.25x with 3.17 accepted/step: killed. The gate may still be right for its original 2k corpus; it is not a long-prompt rule.

## 5. File anchors (current HEAD, read-before-write verified)

- `apps/cli/main.cpp:2165-2167` — `--speculative` is a boolean; no auto tri-state is shipped.
- `apps/cli/main.cpp:788-922` — fused greedy chain (the path measured).
- `apps/cli/main.cpp:924-1070` — non-fused greedy path.
- `prompt_lookup_draft` (`apps/cli/main.cpp`, static helper) — the five-token-window drafter whose yield grows with history.
- `idea-invented-auto-spec-lookup-gate-keep.md` — the gate recipe and its 2k keep; corrected here for long prompts.
- `idea-invented-specation-auto-probation-keep.md` / `idea-invented-auto-spec-2k-measured.md` — the probation cost the gate targeted.
- `idea-invented-spec-at-8k-measured.md`, `idea-invented-code-8k-spec-sparse-measured.md` — the 8k periodic/code rows this cycle completes with prose.

## 6. Reimplementation recipe

1. Build the prose prompt from the distinct-docs corpus (see §1) and sanitize quotes:
   ```powershell
   $all = (Get-Content "$env:TEMP\kilo\corpus_natural_8k.txt" -Raw)
   $p = $all.Substring(0,30000) -replace '"', "'"
   ```
   Confirm the token count with `--context 64 --n 1`; expect `prompt : 8256 tokens`.
2. Interleave the arms (each ~4-5 min of prefill):
   ```powershell
   tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt $p \
     --context 8704 --n 32 --kv f16 --threads 10 --gpu off [--speculative]
   ```
   Expect off 5.78-5.86 t/s and spec 6.91-7.25 t/s with 6 steps / 3.17 accepted per step.
3. Do not add an auto gate: check the spec line's accepted/step; if it is above ~1.5 on prose, a t=0 last-window probe is not a valid predictor.
4. No code ships.

## 7. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56, cycle gate)
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed

prose prompt: 30,000 chars (quote-sanitized) -> 8,256 tokens
8k prose (ctx 8704, n 32, f16, 10 threads, gpu off):
  pair 1: off  5.78 t/s ; spec 7.25 t/s (1.25x, 6 steps, 19 acc, 3.17/step)
  pair 2: spec 6.91 t/s (1.18x, same spec line) ; off 5.86 t/s
git status --short                                -> clean (no code touched)
python aikit.py check aegean-tge                  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
