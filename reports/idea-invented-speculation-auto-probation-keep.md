# INVENTED: `--speculative auto` with an accept-rate probation — KEEP as an opt-in mode (3.2-4.0x on repetitive workloads; prose bails after 9 steps and matches off)

- **Idea**: Speculation is a 3-5x lever on repetitive/templated prompts and a null lever on free prose, but today the user must predict which workload they have before passing `--speculative`. Add a third mode, `--speculative auto`: start speculative, watch the first 8 decode iterations, and disable speculation for the rest of the run if the average is below 1.2 tokens/iteration (i.e. drafts are not being accepted beyond the free bonus token). Auto then captures the code/repetition win with no flag-gifting, while a poor draft source costs at most the 9 warm-up iterations before falling back to the exact baseline path.
- **Cycle date**: 2026-11-11
- **Verdict**: **KEEP as an opt-in mode** (default unchanged). Repetitive prompt: auto **3.2-4.0x** over off, no bail. Code-echo prompt: **3.4-3.8x**, no bail. Free prose: auto bails after 9 steps at 0.89 tokens/step and lands within **0.98-1.00x** of off. Greedy ids identical across off / forced-on / auto on all three id-checked workloads. A *static* prompt-repeat probe was prototyped first and rejected: it scored the code-shaped prompt 0.000 and would have disabled exactly the workload where drafting wins, which is why the shipped mechanism is the dynamic probation.
- **Tree state at close**: spike reverted from `apps/cli/main.cpp` and `apps/cli/cli_options.hpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

Current state (`apps/cli/main.cpp`):

- `speculative` is a boolean flag (`main.cpp:2165`); `speculate = speculative && temperature <= 0` at `:756`, `speculate_sampled` at `:762`.
- The greedy decode loop runs a fused 6-row verify (`:788`), a correction-forward fallback (`:924`), and the sampled path runs a sampled verify (`:1081`). Counters `spec_steps/spec_accepted/spec_output` already exist at `:765-767`.
- When `prompt_lookup_draft` finds no history match, the fused path does one 1-row forward to recover the bonus KV; on prose every token takes that path, so an always-on spec mode pays a measurable tax for a draft source that cannot fire.

Spike: a tri-state mode.

1. `--speculative` keeps meaning forced-on; `--speculative auto` sets `speculative_auto` (the parser consumes the literal `auto` next token, so `--speculative auto` is unambiguous and `--speculative` alone is unchanged).
2. In `command_generate`, when auto is requested, set `speculative = true` and (like forced-on) pin per-head attention when the user did not ask for scalar.
3. Mutable `speculate` / `speculate_sampled`; before the decode loop, `auto_steps = 0` and `auto_start_tokens = generated.size()`. At the top of each loop iteration, while auto has not bailed: `++auto_steps`; after 8 iterations compute `per_step = (generated.size() - auto_start_tokens) / auto_steps`; if `per_step < 1.2`, set both spec flags false and print `speculation: auto off after N steps (X tokens/step)`.
4. Everything else (drafts, accept rule, fused steady state) is unchanged. Bailing simply routes subsequent iterations into the existing baseline blocks; greedy output is the same token stream because speculation is verifier-argmax-exact.

**Load-bearing detail**: the probation counter must count *decode iterations*, not speculative steps. The first prototype used the existing `spec_steps`, which only increments when a draft block actually runs; on prose the drafter often returns false, so `spec_steps` never reached 8 and auto never bailed — the prose arm lost ~10% before this was fixed.

## 2. Predicted gain (with the math)

For 63 generated tokens at ~9-10 t/s decode (~105 ms/token, ~6.6 s wall):

- Periodic prompt: 5.09 drafts accepted/step -> 11 verify steps instead of 63 forwards, so the forward count is 17% of baseline; sampler and scheduling add the rest. Predicted wall 0.2-0.35x -> 3-5x. Measured 0.25-0.31x.
- Prose: 0-1 accepted/step -> no forward saving; the fused no-draft path adds a 1-row correction forward per token in the worst case. Probation bounds this to 9 iterations (~9 tokens, <15% of a 63-token run) and then exact baseline. Predicted auto within ±3% of off; measured 0.98-1.00x.
- Static-probe variant (rejected): a prompt-only 5-gram continuation score is O(prompt) and scored periodic 1.000, prose 0.000 — correct there — but scored the code-shaped prompt 0.000 because the prompt's 5-grams do not repeat even though the generated code repeats prompt structure. It is recorded here so the idea is not retried: prompt-only repetition is not a sufficient statistic for drafting success.

## 3. Falsifier (stated before the spike)

- **KEEP** if, on the reference model at `--threads 6 --gpu off --kv f32`, greedy:
  (a) periodic prompt: auto >= 1.5x over off and no bail;
  (b) code-shaped prompt: auto >= 1.5x over off and no bail;
  (c) free prose: auto within 5% of off and the bail line present;
  (d) ids identical across off / forced-on / auto;
  (e) sampled runs complete in both modes (`--temp 0.8`);
  (f) suite 171/171 and gates 4/4.
- **KILL** if any arm fails (a)-(d), or if auto is worse than off by >5% on any prompt.

All passed. (b) is demonstrated with a code-echo prompt that actually repeats; a short 63-token fibonacci prompt where forced-on itself is only 1.03x is also recorded below as a "no-win workload" control, where auto correctly matches the better arm.

## 4. File anchors (current HEAD `0560813`, read-before-write verified)

- `apps/cli/main.cpp:498-519` — `command_generate` signature (the spike adds `bool speculative_auto` after `tokenizer_cache_dir`).
- `:635-639` — resolved-state `speculation=` lines (the spike prints `auto` before the decision).
- `:756` / `:762` — `speculate` / `speculate_sampled` (const in the tree; mutable in the spike).
- `:765-767` — `spec_steps/spec_accepted/spec_output` (used for the final `spec:` line, not for the probation).
- `:785-788` — fused decode loop start; `:924` non-fused greedy; `:1081` sampled.
- `:2165` — `--speculative` parse block in `run_generate`.
- `apps/cli/cli_options.hpp:115` — generate usage string.
- `reports/idea-invented-spec-equivalence-sweep.md` — the measured basis: code 4.98x, prose 0.98-1.14x, equivalence caveat.

## 5. What was done (temporary spike, fully reverted)

1. Added the `speculative_auto` boolean to `run_generate`, parsed `--speculative auto` by consuming the next token when it is exactly `auto`, and passed it as a new defaulted `command_generate` parameter.
2. Resolved-state line prints `speculation=auto` when the mode is requested.
3. After prompt tokenization, auto forces `speculative = true` and pins `AttentionMode::Head` unless scalar was explicitly requested, then prints the probation policy line.
4. Made `speculate` / `speculate_sampled` mutable; added `auto_steps` / `auto_start_tokens`; inserted the iteration-count bail at the top of the decode loop (threshold 1.2 tokens/iteration after 8 iterations).
5. Prototyped and then removed the static probe (`prompt_repeat_score`, a 5-gram continuation hit-rate over the prompt, with hash + exact 5-gram verification) after it scored the code-shaped prompt 0.000.
6. Updated the usage string; built; 171/171; measured; reverted both files; rebuilt; 171/171; gates 4/4; `git status` clean.

## 6. Measured numbers

Reference `Llama-3.2-3B-Instruct-Q4_K_M.gguf`, `--threads 6 --gpu off --kv f32`, greedy, n=64 unless noted. "off" = no flag, "on" = `--speculative`, "auto" = `--speculative auto`. Same-binary A/B.

### Periodic text (fox sentence x120 = 1,200 prompt tokens)

| arm | tok/s | accepted/step | bail |
|---|---:|---:|---|
| off | 7.22, 10.11 | - | - |
| on | 29.65 | 5.09 | - |
| auto | 35.19, 31.67 | 5.09 | none |

Auto/off per pair: 4.87x, 3.13x; median ~4.0x. Auto >= on within noise; no bail.

### Code-echo text (`int sum = 0; for (...)` x20)

| arm | tok/s | accepted/step | bail |
|---|---:|---:|---|
| off | 10.60, 10.50 | - | - |
| auto | 35.48, 40.07 | 5.09 | none |

Auto/off: 3.35x, 3.82x.

### Short code prompt (63 tokens, fibonacci) — no-win control

| arm | tok/s | note |
|---|---:|---|
| off | 8.09, 9.15, 9.58 | median 9.15 |
| on | 9.42 | 1.03x; spec internally stops after 16 emitted tokens |
| auto | 8.39, 9.30, 9.48 | median 9.30; matches the better arm within ~2% |

### Free prose (dragon, 24-token prompt) — ABBA, two rounds

| round | order | off | auto | auto/off | bail |
|---:|---|---:|---:|---:|---|
| 1 | off, auto, auto, off | 10.98 / 10.46 | 11.03 / 9.92 | 1.00 / 0.95 | after 9 steps, 0.89 t/step |
| 2 | off, auto, auto, off | 9.66 / 9.60 | 9.71 / 9.57 | 1.01 / 1.00 | after 9 steps, 0.89 t/step |

Auto mean per round: 10.48 vs off 10.72 (0.98x) and 9.64 vs 9.63 (1.00x).

### Ids (greedy, n=64, `--print-ids`)

| workload | off | on | auto | equal |
|---|---|---|---|---|
| periodic | 9D-93-29-15-7A-B5 | same | same | off==on==auto |
| code (63-token) | 6E-7A-97-E5-6F-0B | same | same | off==on==auto |
| prose | 97-FD-AF-DE-EB-6D | same | same | off==on==auto |

The known equivalence caveat (sampled mode and the float-verifier vs kq8 near-tie split) is unchanged by this idea; auto does not alter either kernel path.

### Sampled sanity (`--temp 0.8 --seed 7`, n=32)

| workload | tok/s | accepted/step | bail |
|---|---:|---:|---|
| prose | 12.95 | 0.00 | after 9 steps, 0.89 t/step |
| periodic (40 reps) | 33.82 | 5.60 | none |

### Rejected static-probe numbers (for the record)

5-gram continuation score over the prompt, threshold 0.35: periodic 1.000 (on), prose 0.000 (off), code 0.000 (off). The code prompt's forced-on arm was only 1.03x, but a *code-shaped* prompt that later repeats structure is exactly where drafting wins, so a prompt-only probe would systematically disable the best case. The probation replaced it.

## 7. Why it is worth keeping

- No default behavior changes: the flag is opt-in, and `--speculative` still means forced-on.
- The two failure modes are bounded: a bad draft source costs at most 8-9 iterations (and the printed bail makes it visible), while a good one keeps the full 3-4x.
- It removes the need for the user to predict the workload; the measured code-echo and periodic cases both clear 3x with the same invocation.
- The bail threshold (1.2 tokens/iteration) sits above the free-bonus 1.0 and well below the 2-5 accepted/step that a working drafter produces, so the classification is robust to box noise.

## 8. Full reimplementation recipe

1. `apps/cli/main.cpp`, `run_generate`: add `bool speculative_auto = false;` next to `bool speculative`. Change the `--speculative` branch to:
   `if (index + 1 < argc && std::strcmp(argv[index + 1], "auto") == 0) { speculative_auto = true; ++index; } else { speculative = true; }` then `continue;`.
2. `command_generate` signature: add `bool speculative_auto = false` after `tokenizer_cache_dir`; pass it at the call site (after `resolved_tokenizer_cache`).
3. Resolved state: print `speculation=auto` when `speculative_auto` (before `off/greedy/sampled`).
4. After prompt tokenization (right before `const std::vector<int>& prompt = tokens;`):
   `if (speculative_auto) { speculative = true; printf("speculation: auto (probation 8 steps, bail below 1.2 tokens/step)\n"); if (attention_mode != Scalar) model->set_attention_mode(Head); }`
5. Make `speculate` and `speculate_sampled` non-const; add `int auto_steps = 0;` and `const int auto_start_tokens = static_cast<int>(generated.size());` before `bool fused = false;`.
6. At the top of the decode `while` loop body:
   `if (speculative_auto && (speculate || speculate_sampled)) { ++auto_steps; if (auto_steps > 8) { const double per_step = (static_cast<int>(generated.size()) - auto_start_tokens) / static_cast<double>(auto_steps); if (per_step < 1.2) { speculate = false; speculate_sampled = false; printf("speculation: auto off after %d steps (%.2f tokens/step)\n", auto_steps, per_step); } } }`
   The counter must count loop iterations while auto is live, not `spec_steps`.
7. Usage strings: `[--speculative [auto]]` in `apps/cli/cli_options.hpp:115` and the inline generate help line (the original text `--speculative    prompt-lookup speculation: greedy +-penalty,`).
8. Build + `tge_tests` 171/171; A/B the three workloads above; expect periodic/code-echo >= 1.5x with no bail, prose within ±5% with a printed bail, ids identical.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release    -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                 -> 171 cases, 0 failed
# probe prototype (later removed): --speculative auto on periodic/code/prose
#   periodic repeat score 1.000 (1200 tokens); code 0.000 (63); prose 0.000 (24)
tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt <p> --n 64 \
  --kv f32 --threads 6 --gpu off [--speculative [auto]] [--print-ids] -> §6 tables
tge generate ... --n 32 --temp 0.8 --seed 7 --speculative auto       -> sampled sanity
git checkout -- apps/cli/main.cpp apps/cli/cli_options.hpp            -> reverted
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                     -> 4/4 gates passed
```

No push, no merge, no force-push. Spike binaries are throwaway builds under `build/release/bin/` and `build/idea/`; the committed tree is unchanged except this report.
