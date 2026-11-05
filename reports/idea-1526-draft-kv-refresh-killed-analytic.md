# IDEA-1526: Decode-Time Draft-KV Refresh with Adaptive Rescoring — KILLED WITHOUT SPIKE (no decay; no draft KV)

- **Idea ID**: IDEA-1526 (dossier `paper-2610-31098.md`, RIFT: prefill/recent draft-KV budget split + decode-time selection refresh with adaptive rescoring; Qwen3.5 drafter holds >98% full-cache acceptance at long context)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: both falsifier arms already fail on today's tree). No branch created, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff), stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 18 after this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Split a draft model's KV budget between prefill-selected entries and recent
generated tokens; as decode proceeds, refresh the selection (rescore and swap
entries) on an adaptive schedule tied to quality drift. Value: long-output
draft acceptance stops decaying with output length. Scope is inference-time
policy only (compressibility KD excluded).

## 2. Predicted gain — and why the premises fail here

Two premises, both false on this tree:

- (a) Decay premise: draft acceptance decays with output length because a
  FIXED prefill-selected budget goes stale. Our drafters are model-free
  token-id retrieval over the FULL growing history (spans, bigram, gapped,
  SKIP, all computed per step with no budget) — match opportunities grow
  monotonically with length. Measured: 47-token code output 3.45/step
  (38/11); 127-token continuation of the same prompt **3.48/step (94/27)**,
  deterministic 2/2, ids identical. Acceptance rises, not decays.
- (b) Carrier premise: a draft-model KV selection exists to refresh. It does
  not: grep over `src/` and `apps/cli/main.cpp` for `draft.*KV|KV.*draft|
  draft_kv` finds zero draft-cache state (the single hit is the target
  model's bonus KV comment). Sessions share weights, never draft tables.
  There is no budget to split, nothing prefill-selected, nothing to rescore.

## 3. Falsifier (stated; dossier gives contract + failing-first tests, no one-line bar)

Kill if (a) long-output acceptance ≥ short-output acceptance at matched
settings (no decay to fix), or (b) no draft-KV budget carrier exists in the
tree. Both trip: (a) 3.48 ≥ 3.45 with identical ids; (b) zero draft-KV state
by grep.

## 4. File anchors (current `master`, read but untouched)

- `apps/cli/main.cpp` — the four shipped drafters (pure functions of
  per-request `history`; §4 of the 1542 report) + `spec :` accounting.
- `src/models/qwen2.cpp` — the only KV caches are the TARGET model's
  (f32/f16/i8); no draft-model cache exists.
- Dossier contract that cannot attach: CLI `--draft-kv-budget`,
  `refresh-off reproduces current behavior` (vacuous here — there is no
  on-state), "acceptance-decay" failing-first test (would pass trivially:
  no decay, §6).

## 5. What was done (ANALYZE + confirmation runs on unmodified binary — no spike)

1. Re-read the dossier + blueprint against the current tree; identified the
   architecture mismatch (draft-model KV policy vs KV-free retrieval).
2. Grep-proved the carrier absence (zero draft-KV state).
3. Gates: build clean, `ctest` pass; long-output runs (below) on the
   unmodified binary; `aikit.py check` 2/4 (test + secrets pass; hygiene +
   guidelines fail pre-existing).
4. No thinking-set diff owed: same binary, zero behavior change.

## 6. Measured numbers (today's tree, unmodified binary)

Code prompt, Llama-3.2-3B-Q4_K_M, `--speculative --threads 6 --gpu off`:

| output length | steps | accepted | rate |
|---|---|---|---|
| 47 tokens (`--n 48`) | 11 | 38 | 3.45/step (prior baseline, reproduced) |
| 127 tokens (`--n 128`) | 27 | 94 | **3.48/step** (2/2 deterministic) |

Ids byte-identical across the two long runs; the 47-token prefix matches all
prior runs. No decay at 2.7x length — the premise inverts (more history, more
matches). A refresh schedule would pay rescoring latency to fix a non-decay.

## 7. Reimplementation recipe (deliberately NOT recommended — kill-grade)

For the record: would require first inventing a draft-model KV cache (new
subsystem), then budgeting/refreshing it — i.e. building the very drafter
architecture the tree deliberately avoids (retrieval drafts exist precisely to
have zero draft state). Do not build. Revisit only if the engine ever gains a
draft MODEL with its own KV cache AND long-output acceptance is measured
decaying — neither holds.

## 8. Verdict and why

**KILLED without spike.** RIFT solves draft-KV staleness; this tree has no
draft KV and no staleness (acceptance rises with length: 3.45 → 3.48). The
dossier's own combat notes ("net win must be measured end-to-end", "gate
refresh on measured drift so short outputs pay nothing") agree: measured drift
is zero-to-positive, so the gate would permanently disable the refresh. A
permanently-off refresh is not a feature.

## 9. Exact commands run and their results

- Grep draft-KV carrier → zero matches (mechanism has nothing to attach to).
- `cmake --build --preset release` → 54/54; `ctest` → pass.
- `generate` (code prompt, `--n 128`, ×2) → 3.48/step both, ids identical.
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits, no push.
