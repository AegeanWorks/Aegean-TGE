# INVENTED: --kv auto (Context-Adaptive KV Tier) — PROCEED WITH RECIPE (fidelity-identical to manual, short takes f32, long takes f16; one false alarm resolved by interleaving)

- **Idea**: Invented (no dossier; measurement-log check: only line 852 "default stays f32; --kv f16 for long-context" — manual flag, no auto researched). `--kv auto` selects the KV tier on declared context capacity: f16 above 512, f32 at/below (i8 never auto — conditional per the size ladder). Move less where it pays (f16 halves KV traffic where attention is 13–21% of decode at 2k), keep the exact f32 path where it doesn't (short histories: all tiers within 4%, §41).
- **Cycle date**: 2026-11-06
- **Verdict**: PROCEED WITH RECIPE (conditional keep — not committed as code per loop rules; the spike is fully reverted). Every falsifier arm passes: fidelity auto == manual f16 to the digit (KL 2.021e-06, worst 1.609e-04, 1023/1023); short auto-256 == manual f32-256 byte-identical ids; 2048 auto runs the f16 path (load print half=1, 13% attn phases, ids identical to manual arms) within 2% of interleaved manual f16. One false alarm on the way (paired-but-not-interleaved runs showed auto at f32 speed) resolved by interleaving: the box had drifted between windows, both arms moving together. Methodology note stands: on this box, pair-interleaved or re-baseline within the window — adjacent pairs across 10-minute spans can lie by 13%.
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all 5 spike files reverted with empty diff verified. `master` ahead of `origin/master` by 56 before this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked)

Pure rule + plumbing, no kernel work. `Qwen2::auto_kv_half(context_limit)`: declared capacity (0 = the loader's 4096 default) above `kAutoKvHalfContext = 512` → f16 else f32; i8 excluded by design. Threshold grounds: §41 ladder (128: all within 4%; 512: f16 best, i8 worst) + 2048 closure (f16 best, ids parity). Resolution at the four `--kv` parse sites (generate/fidelity/continuous/sessions) post-parse, before the command call — no signature changes, no load-path changes, resolved bools flow through the existing `kv_half/kv_q8` plumbing. Fidelity reference arm untouched (stays f32); candidate resolves auto the same way. Known edge (documented, accepted): budget-shrink can resolve f16 for a run that ends up short — harmless (f16 priced at 1.7e-6 with top-1 held; short f16 within 4% of f32).

## 2. Predicted gain (with the math)

At 2048 history: +9–16% decode (closure ladder: f32 8.26 → f16 9.26), minus nothing — same binary path as manual f16. At ≤512: 0% by construction (selects f32, identical path). Load: identical or smaller (same buffer count, half bytes when f16). Cost: one integer compare per invocation.

## 3. Falsifier (stated) — all arms pass

Kill if auto selects differently than manual for the same context (mapping bug), or any ids divergence vs manual same-format runs at matched history, or short (≤512) doesn't take the exact f32 path, or 2048 doesn't reproduce manual f16 within noise. Results: fidelity auto row == manual f16 row digit-for-digit (§6.1); short auto-256 ids == manual f32-256 ids byte-identical (§6.2); 2048 auto runs f16 (tier prints + 13% attn phases + identical 62 ids) at 8.38 mean vs interleaved manual f16 8.52 mean — within 2% (§6.3). Thinking set fluent on the auto tier (§6.4).

## 4. File anchors (master = pre-spike; spike reverted)

- `include/tge/models/qwen2.hpp:182-187` (kv_half/kv_q8 accessors — the rule lived beside them as `kAutoKvHalfContext` + `auto_kv_half` decl, reverted).
- `src/models/qwen2.cpp:25` (`kDefaultContext = 4096` — the 0-rule), `:346-369` (context resolution + budget shrink), `:465-478` (kv allocator switch — untouched; resolved bools flow in).
- `apps/cli/main.cpp` (4 `--kv` parse sites: generate ~2069, fidelity ~2506, continuous ~2708, sessions ~2745; resolve points before the 4 command calls; error strings + reference-arm exclusion).
- `tests/test_attention.cpp` (model-policy test site; spike test `auto_kv_half_threshold` green, reverted).
- Grounds: measurement-log line 852 (manual guidance this automates), §§41/2048-closure (threshold + ladder), §29 (f16 1.7e-6 price).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-kv-auto` (zero commits ever made on it; deleted at close).
2. Spike (5 files): header rule + constexpr; impl using file-local `kDefaultContext`; 4 CLI sites (`kv_auto` local, "auto" branch, post-parse resolve, error strings); rule unit test (0/1/511/512/513/1024/4096/32768 + constexpr check). Two temporary tier/load debug prints added mid-cycle during the false-alarm investigation, also reverted.
3. `cmake --build --preset release` clean (54/54, each rebuild); `ctest --preset release` pass; `tge_tests.exe` 142/142 (141 + 1 new).
4. Box health gate: wall 62.45 → 60.79 bracketing (healthy throughout).
5. Fidelity pair (0.5B-Q8_0, arch corpus, ctx 1024, 6 threads, gpu off): manual f16, then auto (label confirms ", f16 kv").
6. Short pair (0.5B story, --context 256): manual f32, then auto — ids diffed.
7. Long triple + interleave (Llama-3B, 2050-tok fox, --context 4096): auto R1, manual f16 paired, auto R2 (false alarm) → interleaved f16/auto/f16/auto (§7), ids + phases captured throughout.
8. Thinking set on auto (default context → f16): reasoning + haiku, fluent with ids recorded.
9. `python aikit.py check aegean-tge` → 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing).
10. Reverted all 5 files (`git checkout --`, empty diff verified), `git checkout master`, `git branch -D spike-kv-auto`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141.

## 6. Measured numbers

6.1 Fidelity (1023 positions, ctx 1024, Q8_0 0.5B) — auto and manual f16 identical to the digit: KL mean 2.021e-06, worst 1.609e-04, max|diff| 8.177e-02, top-1 1023/1023 (31.09 s vs 32.02 s wall — noise). The ", f16 kv" candidate label confirms the candidate ran f16 end to end.

6.2 Short (story, 4-tok prompt, ctx 256, 15 ids): auto-256 ids == manual f32-256 ids byte-identical (`11 1052 572 … 13`, both runs) — the exact f32 path, zero behavior delta where the tier doesn't pay.

6.3 Long (2050 tokens, ctx 4096, 62–63 tok decode): false-alarm sequence then resolution —
paired (non-interleaved): auto 50.5–51.3/8.07×2 vs manual f16 54.6/9.24 → apparent 13% deficit with f16 phases (13% attn) and f16 tier prints (resolve + `load kv: half=1`) — contradictory, investigated, not shipped as a verdict;
interleaved f16/auto/f16/auto: f16 52.6/8.07 + 50.5/8.96 (mean 8.52) vs auto 50.7/8.12 + 52.8/8.63 (mean 8.38) — within 2%, phases 12–13% both arms, 62 ids identical across all four runs (and identical to the closure round's f32/f16/i8 ids). The box had drifted between the paired windows (manual f16 itself fell 9.24 → 8.07); interleaving reunites the arms. Verdict stands on the interleaved data.

6.4 Thinking set (spike binary, `--kv auto`, default context → f16): reasoning answer fluent (32 ids `720 16533 … 40120` recorded); haiku fluent ("Nature's peaceful hush / A haiku is" + closing line). No degenerate output on the auto-selected tier.

## 7. Full reimplementation recipe (keep-grade — rebuild exactly this)

1. `qwen2.hpp`, beside `kv_half()`: `static constexpr int kAutoKvHalfContext = 512;` + `static bool auto_kv_half(int context_limit);` with the §1 doc comment.
2. `qwen2.cpp`, near the context constants: `bool Qwen2::auto_kv_half(int context_limit) { const int declared = context_limit > 0 ? context_limit : kDefaultContext; return declared > kAutoKvHalfContext; }` (uses the file-local `kDefaultContext`, no new constants).
3. `main.cpp`, at each of the 4 `--kv` value sites: `bool kv_auto = false;` local, `"auto"` branch (sets `kv_auto = true`, clears half/q8), post-parse `if (kv_auto) { kv_half = Qwen2::auto_kv_half(<that command's context var>); kv_q8 = false; }`, error string `f32|f16|i8|auto`. Reference arm (fidelity line ~2302) stays f32 — do not touch.
4. `tests/test_attention.cpp`: `auto_kv_half_threshold` (boundaries 0/1/511/512/513/1024/4096/32768 + constexpr check).
5. Gate: build 54/54, 142/142 tests, fidelity auto == manual f16 digit-for-digit (the §6.1 triplet), short auto-256 == manual f32-256 ids, 2048 auto within noise of INTERLEAVED manual f16 with identical ids. Do not special-case i8 into auto (size-ladder conditional stands). Do not resolve on actual prompt length (load-order chicken-and-egg; declared capacity is the documented rule with the budget-shrink edge noted in §1).

## 8. Verdict and why

**PROCEED WITH RECIPE.** The cheapest win on record this cycle: one integer compare buys the measured +9–16% long-context decode with zero short-context delta, fidelity-priced numerics, and byte-identical behavior to the manual flags on every arm. The only drama was methodological (non-interleaved pairs lying by 13%), and it leaves a standing rule for this box: interleave or re-baseline within the window. Revival trigger: none needed — the recipe is complete; the remaining decision (ship as default vs flag) is a product call, and the data says flag-first (f16 changes numerics microscopically; operators who need bit-identical f32 keep it Explicitly).

## 9. Exact commands run and their results

- `git checkout -b spike-kv-auto` → branch (zero commits, deleted at close).
- 5-file spike (§5.2) + 1 test + 2 temporary debug prints (all reverted).
- `cmake --build --preset release` → 54/54 (every rebuild); `ctest --preset release` → pass; `tge_tests.exe` → 142/142 on spike, 141/141 after revert.
- `bench mem` wall 62.45 → 60.79 (healthy bracketing).
- Fidelity manual f16 == auto to the digit (§6.1); short auto-256 == manual f32-256 ids (§6.2); long paired showed the false alarm, interleaved resolved it (§6.3); thinking set fluent (§6.4).
- `python aikit.py check aegean-tge` → 2/4 (pre-existing hygiene + guidelines fails).
- `git checkout -- <5 files>` → empty diff; `git checkout master`; `git branch -D spike-kv-auto`; final master gates green; report committed; no push.

(End of file.)
