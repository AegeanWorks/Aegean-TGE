# IDEA-1542: Lossless Ngram Speculative Draft plus Priced KV Types — SUPERSEDED (both halves ship; sub-i8 KV residue named for the KV track)

- **Idea ID**: IDEA-1542 (dossier `kilo-pass3-2026-11-03.md`: llama.cpp ngram-cache draft family + per-type KV pricing; register notes it distinct from 1502's adaptive gate)
- **Cycle date**: 2026-11-06
- **Verdict**: SUPERSEDED — not killed, not kept: today's tree already implements and prices both halves. The draft family is saturated (four shipped drafters + the just-measured trie), KV f16/i8 are priced and kept with receipts. The only genuine delta — sub-i8 KV tiers (q4_0 and below) — is named below as a KV-track cycle (with 1516/1515), not this idea. No spike built (nothing to build that isn't a repeat); no branch, no code touched.
- **Tree state at close**: `git status` clean (no diff), stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 17 after this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Greedy n-gram draft (constant-memory hash pool shared across slots) under a
lossless verifier guarantee, plus per-KV-type fidelity pricing (KV f16/q8_0 at
KL ~1e-6–1e-3 with top-1 intact) as the template for `bench continuous` /
`--speculative` wins. "Must be true": greedy ngram draft preserves ids vs
sequential; KV q8_0/q4_0 price inside budget; pinned per-head verifier schedule
holds.

## 2. Predicted gain — and why the tree already collects it

- Draft half: llama.cpp's ngram-cache/simple/map/k/map-k4v/mod are order-k
  majority/hash lookups over history. Today's tree fields FIVE such mechanisms:
  exact spans (all lengths), order-1 bigram statistics with exact counts,
  one-substitution gapped windows, period-1/2 SKIP-lock, plus last cycle's
  exact order-2/3 majority-vote trie — measured at 3.45→3.18/step (lost) on
  the code prompt, ties elsewhere, ids identical (`59e22f6`). Every
  ngram-draft structure in the dossier's list maps onto one of these; no
  untested structure remains (order-4+ contexts are sparser than the order-2/3
  contexts that already abstain).
- KV half: f16 priced (KL **1.658e-6**, top-1 497/497, kept) and i8 priced
  (KL **3.2e-4**, top-1 31/31, kept) — inside the dossier's budgets (f16 ≤
  1e-5, q8 ≤ 1e-3). The verifier is greedy and pinned (per-head schedule),
  so ids cannot diverge by construction — confirmed byte-identical across
  every speculative run on record.

## 3. Falsifier (from the dossier) — does not trip

Ids diverge vs fidelity oracle (never: verifier-gated, identical in all
recorded runs), KL over budget (f16 1.658e-6 ≤ 1e-5 ✓, i8 3.2e-4 ≤ 1e-3 ✓),
or no win on `bench continuous`/`generate --speculative` (wins ship: story
1.66x, periodic 6.00/step ceiling, code 3.45/step — re-confirmed today, §6).
Every arm passes; there is nothing to kill, only prior art to record.

## 4. File anchors (current `master`, read but untouched)

- Drafters: `apps/cli/main.cpp` (`skip_locked_draft`, `prompt_lookup_draft`,
  `bigram_chain_draft`, `gapped_draft`) + reports `59e22f6` (trie) and
  `5e19e9f` (rolling-hash kill).
- KV pricing: `src/models/qwen2.cpp` (`kv_half_`/`kv_q8_` store + read paths);
  CLI `--kv f32|f16|i8`; `docs/measurement-log.md` H1 fidelity block (f16
  1.658e-06/497, i8 3.2e-04/31) and H3 78%-vs-f16 decode note.
- Continuous/spec: `bench sessions`, `bench continuous`, `--speculative` with
  `spec : steps/accepted/step` accounting.

## 5. What was done (ANALYZE + one confirmation run — no spike)

1. Mapped every dossier sub-mechanism to shipped/measured counterparts (§2).
2. Re-confirmed the shipped win on current master (code prompt, Llama-3B):
   3.45/step, ids identical to all prior runs (§6).
3. Gates: build clean, `ctest` pass, `aikit.py check` 2/4 (test + secrets
   pass; hygiene + guidelines fail pre-existing).
4. No thinking-set diff owed: same binary, zero behavior change.

## 6. Measured numbers (today, unmodified master)

Code prompt, Llama-3.2-3B-Q4_K_M, `--speculative --threads 6 --gpu off`:
`spec : 11 steps, 38 accepted (3.45/step)`; `decode 47 tokens, 21.22 t/s`;
ids `356 1044 ... 320 396` — byte-identical to the spike/baseline runs of the
1387 cycle. Cited receipts: KV f16 KL 1.658e-6 (497/497), i8 KL 3.2e-4 (31/31).

## 7. Open residue (for the KV track, NOT this idea)

Sub-i8 KV tiers (q4_0 and below) are unpriced on this tree — no store path, no
KL receipt, no decode rate. That work belongs to IDEA-1516 (KIVI 2-bit,
falsifier: KL > 1e-3 / top-1 < 490/497 / decode ≤ i8 → kill) possibly paired
with 1515 (mixed-precision budget), as a dedicated KV-tier cycle with its own
fidelity-first staging. This report must not be read as pricing them.

## 8. Verdict and why

**SUPERSEDED.** An idea whose every checkable claim the tree already satisfies
— with receipts — is prior art, not a task. Building another ngram-draft
variant would repeat the 1387 experiment with a weaker (lossy or longer-sparse)
signal; pricing f16/i8 again would re-pay settled fidelity runs. The loop's
honest output here is the map (§2), the confirmation (§6), and the carved
residue (§7). Revisit only the residue, under the 1516 falsifier.

## 9. Exact commands run and their results

- `cmake --build --preset release` → 54/54; `ctest` → pass.
- `generate` (code prompt, Llama-3B, `--speculative`) → 3.45/step, ids match.
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits, no push.
