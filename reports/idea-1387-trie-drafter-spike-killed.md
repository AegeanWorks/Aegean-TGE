# IDEA-1387: Online Majority-Vote Trie Drafter — SPIKED, MEASURED, KILLED (acceptance 3.45 → 3.18)

- **Idea ID**: IDEA-1387 (dossier `scout-2026-11-03-kilo-4target.md`: "Online-learned trie/PPM retraining-free drafter"; register confirms the gap: no trie or PPM in src; prompt_lookup spans/bigram/gapped ship in `main.cpp`)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED by its falsifier — acceptance fails to beat prompt_lookup on every prompt (code 3.45→3.18 deterministic, story 1.00→1.00, periodic 5.33→5.33). Spike fully reverted; only this report committed.
- **Tree state at close**: `git status` clean (no diff), spike branch created and deleted with nothing committed, stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 14 after this report. Nothing pushed (human gate).

## 1. Mechanism

Online Boyer-Moore majority-vote trie over token ids, learned incrementally as
history grows: per bigram (a,b) and unigram (b) context keeps one (best-next,
count) pair (streaming vote, one entry per seen context, hard caps 8192/2048 so
the structure stays L3-resident). Draft chains up to 6 tokens with bigram →
unigram backoff, all-or-nothing like every shipped drafter, tried ONLY after
skip_locked and prompt_lookup fail or miss the d0 pre-check — so any acceptance
gain would be purely marginal. Same d0 pre-check, same verifier; ids identical
by construction (verifier-gated, greedy argmax rule untouched).

## 2. Predicted gain (with the math)

Gap: bigram_chain is order-1 with a full-history rescan per draft position;
exact spans need `want` continuations after the match, which the fresh tail
often lacks. The trie chains single steps from older statistics regardless of
tail length — the natural next rung for code with recurring-but-varied n-grams
(`for (`, `std::vector<`, loop bodies). Each accepted draft token saves one
correction forward; at 1.00/step → ~2x fewer sweeps is the optimistic shape.
Falsifier (dossier): acceptance fails to beat prompt_lookup, or added L3
pressure, on story + code prompts (plus periodic as ceiling sanity) → kill.

## 3. Falsifier (from the dossier, operationalized)

- Code prompt acceptance (tokens/step) ≤ prompt_lookup baseline → kill.
- Story acceptance ≤ baseline, or periodic ceiling disturbed → kill.
- Any ids divergence between binaries → kill (drafter contract).
- Wall-clock regression attributable to trie CPU/L3 → kill.

## 4. File anchors (current `master`, read before writing)

- `apps/cli/main.cpp:472-475` — drafter forward declarations; `:1770-1815`
  bigram_chain (order-1, rescans history per position); `:1826-1858` gapped
  (one-substitution windows); `:1866-1902` skip_locked; `:1915-1953`
  prompt_lookup (spans → bigram → gapped, 6-deep with 5-deep fallback);
  `:477` `command_generate` (history at `:672`, greedy verify `:860-999`,
  `spec : N steps, M accepted (X.XX/step)` report at `:1188-1193`).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-1387-trie-drafter` (no commit ever made on it).
2. `main.cpp` only (+119 lines): `<cstdint>` + `<unordered_map>` includes;
   `TrieDrafter` struct (Vote/bigram/unigram/trained/caps/key/vote/observe/
   train/draft); `TrieDrafter trie_drafter;` beside `history`; `train(history)`
   before the draft block (incremental O(new) triples); last-resort draft
   attempt after prompt_lookup with the identical d0 pre-check.
3. `cmake --build --preset release` clean; `ctest` pass; 141/141 (CLI is not
   unit-tested; drafter-only precedent: ids + acceptance are the gates, no
   fidelity owed — no numerics touched, verifier decides everything).
4. Llama-3.2-3B-Q4_K_M, `--speculative`, 3 prompts × best-of-3 (spike), saved
   spike binary, stashed, rebuilt baseline, repeated all runs.
5. Disambiguation run: code prompt single-threaded on both binaries
   (deterministic FP reduction order — rules out threaded near-tie flips).
6. Reverted (`git checkout --`), verified empty diff, deleted branch. Final
   gates on master: build clean, `ctest` pass, `aikit.py check` 2/4 (test +
   secrets pass; hygiene + guidelines fail pre-existing).

## 6. Measured numbers

Llama-3.2-3B-Q4_K_M, `--speculative --gpu off`, warm page cache. Deterministic
3/3 within each binary on every prompt.

| prompt | baseline acceptance | spike acceptance | ids |
|---|---|---|---|
| periodic (`red`×40, `--n 16`, 6T) | 5.33/step (16/3) | 5.33/step (16/3) | identical (`2579`×16) |
| story (robot on Mars, `--n 32`, 6T) | 1.00/step (1/1) | 1.00/step (1/1) | identical (31 ids) |
| code (loop boilerplate, `--n 48`, 6T) | **3.45/step (38/11)** | **3.18/step (35/11)** | identical (47 ids) |
| code, single-threaded (determinism check) | **3.45/step (38/11)** | **3.18/step (35/11)** | identical |

Kill arm: the opportunity prompt accepts FEWER drafts with the trie (38→35,
deterministic incl. single-threaded, so not a threaded near-tie flip — the
trie genuinely fires and displaces better drafts). Mechanism of harm: on steps
where prompt_lookup's draft misses d0, the trie sometimes matches d0 with a
statistically-plausible-but-worse continuation; the verifier then burns a full
6-row sweep for fewer net tokens than the baseline's correction-forward path
mix. Neutral prompts tie (trie abstains where there is nothing to learn, or
proposes what bigram already covers). Wall clock is noise-dominated and NOT
claimed: 6T code 18.4 (base) vs 22.0 (spike) t/s goes the "wrong way" relative
to acceptance (−3 accepted should mean more work, yet faster) — ±20% box
variance per the machine note; acceptance is the clean metric and it went down
deterministically.

L3 pressure arm: moot — maps capped at 8192/2048 entries (actual use: tens of
entries, a few KB); kill rests on acceptance alone.

## 7. Full reimplementation recipe (kill-grade — rebuild only to re-verify)

1. Includes: `<cstdint>`, `<unordered_map>`.
2. Before `command_generate`, define `struct TrieDrafter` with `Vote{next,
   count}`, `bigram`/`unigram` maps, `trained` index, caps 8192/2048;
   `bigram_key(a,b) = (uint64)a<<32|uint32b`; Boyer-Moore `vote()`;
   `observe(a,b,next)` with cap-guarded emplace; incremental `train(history)`
   from `max(trained,2)`; all-or-nothing `draft(history,out,need)` chaining
   bigram→unigram backoff (need 1..6, history ≥ 2).
3. In `command_generate`: `TrieDrafter trie_drafter;` next to `history`;
   `trie_drafter.train(history);` before the greedy draft block; after the
   prompt_lookup block add `if (!used_skip && !first_ok &&
   trie_drafter.draft(history, draft, 6))` with the identical d0 pre-check
   (both penalty branches).
4. Gate: the three prompts above on Llama-3B-Q4_K_M, `--speculative`,
   best-of-3 + single-threaded code pair — expect 3.45→3.18 on code, ties
   elsewhere, ids byte-identical everywhere.

## 8. Verdict and why

**KILLED by measurement.** The learned trie is a real mechanism that fires and
hurts: where exact spans and bigram statistics already extract everything the
history offers, order-2 majority votes add no acceptance on neutral text and
actively displace better drafts on structured text (−3 accepted over 11 steps,
deterministic). The shipped chain (SKIP-lock → spans → bigram → gapped) already
saturates the drafter headroom on this model; a statistical fifth source needs
evidence the first four miss systematically, and the code prompt shows the
opposite. Do not revisit without a prompt class where all four shipped drafters
abstain AND order-2+ contexts recur — nothing in the dossier or this box
demonstrates one.

## 9. Exact commands run and their results

- `git checkout -b spike-1387-trie-drafter` → new branch.
- `main.cpp` spike (+119: includes, struct, hook, draft attempt).
- `cmake --build --preset release` → 54/54; `ctest` → pass, 141/141.
- Spike `generate --speculative` ×3 prompts × best-of-3 (§6 spike column).
- Saved spike exe; `git stash push -- apps/cli/main.cpp`; rebuilt baseline.
- Baseline same 9 runs (§6 baseline column); single-threaded pair both binaries
  (35 vs 38, ids identical).
- `git stash pop`; `git checkout -- apps/cli/main.cpp` → diff empty;
  `git checkout master`; `git branch -D spike-1387-trie-drafter`.
- Final: build clean, `ctest` pass, `aikit.py check` 2/4 (pre-existing fails).
- No push.
