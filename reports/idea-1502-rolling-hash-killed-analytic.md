# IDEA-1502: Rolling-Hash Speculative Memory with Adaptive Gate — KILLED WITHOUT SPIKE (exact version already lost; no cross-request carrier)

- **Idea ID**: IDEA-1502 (dossier `kilo-sweep-2026-11-03-b2.md`: "Rolling-hash speculative memory with adaptive gate", ml-explore/mlx-lm PR 1297 — hash(ngram)→next lossy table + per-request state, verifier-corrected byte-identical, cross-request reuse; dossier section truncated mid-source-note, so the falsifier below is stated operationally)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: both falsifier arms already fail on today's tree). No branch created, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff), stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 15 after this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Lossy hash(ngram)→next-token table with per-request state, consulted as a
draft source under an adaptive gate (consult only when recent acceptance is
high), verifier-corrected so output stays byte-identical, with the table reused
across requests. Claimed value: cheaper-than-exact memory of recurring n-grams
plus cross-request amortization of the learning.

## 2. Predicted gain (with the math) — and why it is ≤ the already-measured exact version

Decompose the proposal into (a) learned n-gram→next drafting and (b)
cross-request reuse + lossy hashing + gating:

- (a) was spiked exactly last cycle (IDEA-1387, report `59e22f6`): an EXACT
  online majority-vote trie (strictly more signal than any lossy hash — hash
  collisions can only merge unrelated contexts and add wrong proposals while
  saving ~ns already spent in O(1) lookups). Result: acceptance 3.45→3.18 on
  the code prompt (deterministic incl. single-threaded), ties elsewhere, ids
  identical. Acceptance is monotone in table exactness: a lossy table cannot
  beat the exact table that already loses. Ceiling for (a): ≤ 3.18/step where
  prompt_lookup holds 3.45.
- (b): lossy hashing saves nothing measurable (exact lookups were already
  unmeasurable in wall clock — the 1387 wall deltas were ±20% box noise going
  the "wrong way" vs acceptance); the adaptive gate saves CPU only when the
  table is useless, which cannot repair an acceptance deficit (the gate changes
  when the table is consulted, never what it proposes); cross-request reuse
  (§4) has no carrier in this engine.

## 3. Falsifier (stated operationally; dossier source-note truncated)

Kill if (i) an exact learned n-gram table fails to beat prompt_lookup on the
same story/code/periodic prompts — established 2026-11-06 (3.45→3.18 code,
1.00→1.00 story, 5.33→5.33 periodic, ids identical); or (ii) no cross-request
draft carrier exists in the tree. Both trip: (i) by the 1387 report's
deterministic numbers, (ii) by grep below.

## 4. File anchors (current `master`, read but untouched)

- `apps/cli/main.cpp` — all four shipped drafters are pure functions of the
  per-request `history` vector (`skip_locked_draft`, `prompt_lookup_draft`,
  `bigram_chain_draft`, `gapped_draft`); the 1387 spike confirmed all draft
  state is call-local. Grep for persistent/cross-request draft state
  (`static.*draft`, `global.*draft`, `shared.*draft`, `cross.*request`,
  `draft.*cache`, `table.*draft`): **zero matches**.
- Sessions (`bench sessions`, `bench continuous`) share weights, never draft
  tables; each sequence owns its history. There is nothing for a cross-request
  table to attach to without inventing a new global structure (with staleness
  and locking costs the dossier never prices).
- `reports/idea-1387-trie-drafter-spike-killed.md` — the exact-version numbers
  this kill rests on.

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Re-read the dossier section (noting truncation) and the 1387 report against
   the current tree; confirmed the learning half is mechanism-identical to the
   just-killed exact trie, minus exactness.
2. Grep-proved the reuse half has no carrier (zero cross-request draft state).
3. Gates for the record: `cmake --build --preset release` clean, `ctest` pass,
   `python ../../aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene
   + guidelines fail pre-existing — C++-only repo, README drift untouched).
4. No thinking-set diff owed: same binary throughout, zero behavior change.

## 6. Measured numbers (cited, not re-run — same tree, same binaries as last cycle)

From the IDEA-1387 cycle (Llama-3.2-3B-Q4_K_M, `--speculative --gpu off`,
deterministic 3/3 within binary, single-threaded pair to exclude
threaded near-tie flips):

| prompt | prompt_lookup baseline | + exact learned trie |
|---|---|---|
| code (`--n 48`) | 3.45/step (38/11) | 3.18/step (35/11) |
| story (`--n 32`) | 1.00/step | 1.00/step |
| periodic (`--n 16`) | 5.33/step | 5.33/step |

Ids byte-identical in all 20 runs. A lossy hash table's acceptance is bounded
above by the exact table's 3.18/step on the opportunity prompt — below the
3.45 bar by a deterministic margin. No wall-clock claim is made (box ±20%).

## 7. Reimplementation recipe (deliberately NOT recommended — kill-grade)

For the record only: `unordered_map<uint64_t,int>` from rolling hash to next
token + per-request vector, consulted after prompt_lookup under an
acceptance-rate gate, shared across requests via a mutex-guarded global with
generation staleness. Expected outcome per §2/§6: acceptance ≤ 3.18/step on
code (below prompt_lookup's 3.45), ties elsewhere, plus unpriced locking and
staleness surface. Do not build.

## 8. Verdict and why

**KILLED without spike.** The learning half already lost in its strongest
(exact) form on this tree with deterministic numbers one cycle ago; lossy
hashing, gating, and reuse cannot recover an acceptance deficit — collisions
worsen proposals, the gate changes timing not content, and reuse has no
carrier. Revisit only if (a) a prompt class appears where all four shipped
drafters abstain AND order-2+ contexts recur across requests (nothing
demonstrates one), or (b) the engine gains a cross-request state layer that
changes the carrier analysis. Until then this stays killed alongside 1387.

## 9. Exact commands run and their results

- Grep for cross-request draft state → zero matches (mechanism carrier absent).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass.
- `python ../../aikit.py check aegean-tge` → 2/4 (test, secrets pass; hygiene, guidelines fail pre-existing).
- No spike branch, no edits (`git status` clean apart from the prior ahead count), no push.
