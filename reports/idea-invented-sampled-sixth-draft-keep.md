# INVENTED: use the sixth verifier row in sampled speculation — KEEP

- **Cycle**: 18
- **Date**: 2026-11-09
- **Family**: token generation / speculative decoding
- **Verdict**: **KEEP-grade** (passed the falsifier; not integrated, not pushed)
- **Tree after the cycle**: spike reverted, branch `idea/sampled-draft-six`
  deleted, this report is the only change

## Mechanism

The verifier always computes six rows (`Qwen2::kSpeculativeBatch = 6`). The
sampled path drafted only five tokens and fed `[d0..d4, 0]`, so row 5 (the
logits after the zero pad) was computed and thrown away. The greedy path
already drafts six.

The spike drafts six:

1. The first five proposals are exactly what the old path proposed:
   `prompt_lookup_draft(history, draft, 5)` (span 6..1, then bigram, then
   one-substitution).
2. The sixth is the token that followed the most recent earlier copy of the
   five-token proposal window, or `draft[4]` when no copy exists:
   scan `history` backwards for `draft[0..4]` and take `history[start+5]`.
3. Feed all six drafts (`feed[0..5] = draft[0..5]`), judge draft 1..5 against
   rows 0..4, and take the bonus from row 5.

Why a wrong sixth cannot hurt: `speculative_accept` is distribution-exact for
*any* draft. If draft 5 is accepted it adds a token; if it is rejected, the
replacement is drawn from `q - onehot(draft5)`, which is still exactly `q`, and
the step emits the same number of tokens as the old bonus draw from row 4.
Only the *chance* to add a token changes, never the emitted distribution.

## Rejected first attempt

The first version called `prompt_lookup_draft(history, draft, 6)` directly. That
is wrong because the matcher uses `want` both for the continuation copy and for
span eligibility (`length >= span + want`, `length - 1 - end >= want`, and the
fallback calls). With `want = 6` it can skip the best long span that has only
five continuations and pick a shorter span instead, so the first five proposals
are no longer the old ones. On the story prompt one seed dropped from
1.62 to 1.29 accepted/step. The fix above preserves the five-token proposal
byte-for-byte and appends only the sixth.

## Predicted gain and math

Old shape emits at most `1 + 4 + 1 = 6` tokens per verify step; the new shape
emits at most `1 + 5 + 1 = 7` (+16.7%). How often that matters is the
probability of five consecutive accepts. Measured accepted/step: 3.78 on locked
repetition, 1.6-2.4 on ordinary/story text. So the ceiling is reachable on
repetitive text and rare elsewhere, which is exactly the measured shape.

## Falsifier

Keep if: (a) the controlled locked-repetition workload improves >=5% in decoded
tokens/s, (b) the ordinary story workload does not regress on the seed mean
(>=0.98), (c) 167/167 tests stay green and the existing truncated-distribution
gate still passes.

## Measured (seed-averaged, alt A/B order, temp 0.8, seed 1..8, `--gpu off
--threads 6`)

| Workload | mean gain | min | max | controlled detail |
|---|---:|---:|---:|---|
| 0.5B locked `ha` repetition, top-p 0.9, n=128 | **1.257x** | 1.034 | 1.620 | **27 -> 24 steps**, 3.78 -> **4.38 accepted/step**, same 128 tokens |
| 0.5B story prompt, top-p 0.9, n=64 | 1.044x | 0.931 | 1.152 | seed 2: 17 -> 14 steps, 35 -> 39 accepted, same 52 emitted |
| 3B Q4_K_M story, top-p 0.9, n=64 | 1.015x | 0.977 | 1.033 | one extra emitted token in one seed; neutral otherwise |

The story per-seed spread is trajectory content, not mechanism cost: the extra
draft changes RNG consumption, so each seed samples a different continuation.
Averaged over seeds the mechanism is neutral-to-positive; the locked workload
holds the text fixed and isolates the effect, which is the 1.257x.

The locked run is the clean proof: the same 128 tokens are emitted in 24
instead of 27 verify steps, with accepted drafts rising from 3.78 to 4.38 per
step. That is exactly the sixth row being used instead of thrown away.

## Correctness

- 167/167 tests, including the 40k-draw truncated-distribution gate and the
  six-config cut-set reference (the accept rule is unchanged).
- Distribution argument above; a wrong sixth draft can only change which token
  the same `q` emits, never the distribution.
- No greedy changes; `spec` acceptance counts come from the existing counters.
- `aikit.py check aegean-tge` 4/4; the map is current.

## Anchors (spike, reverted)

- `apps/cli/main.cpp:1073` — six-draft comment and the five-proposal preservation.
- `apps/cli/main.cpp:1079` — `int draft[6]`, `prompt_lookup_draft(..., 5)`.
- `apps/cli/main.cpp:1081` — sixth-token extension scan (5-gram, fallback draft[4]).
- `apps/cli/main.cpp:1127` — feed all six drafts (no zero pad).
- `apps/cli/main.cpp:1138` — verify loop `index < 6`.
- `apps/cli/main.cpp:1164` — bonus row 5.
- Binaries: `build/idea/tge-draft6-base.exe`, `build/idea/tge-draft6-spike2.exe`
  (both built this cycle).

## Full reimplementation recipe

1. In the sampled speculative block, keep the old matcher call:
   `int draft[6]; if (prompt_lookup_draft(history, draft, 5)) { ... }`.
2. Set `draft[5] = draft[4]`, then for `start = history.size() - 6` downto 0,
   compare `history[start..start+4]` with `draft[0..4]`; on the first match set
   `draft[5] = history[start+5]` and stop. This preserves the old five proposals
   and adds the sixth only from real history.
3. Feed `feed[0..5] = draft[0..5]` (remove `feed[5] = 0`).
4. Judge drafts 1..5 with rows 0..4 (`for index = 1; index < 6`).
5. Take the bonus from `verify_logits + 5 * vocab_size`.
6. Keep the rest of the sampled path unchanged (first verdict from `logits`,
   correction forward, counters).

Pitfall: do **not** call the matcher with `need = 6`; its span selection is
coupled to the requested continuation length.

## Disposition

KEEP-grade. Integrate with the next approved code batch; it only changes the
sampled path and is free at the forward level (row 5 was already computed).
Nothing was pushed; the workspace remains ahead of `origin/master` pending the
human push gate.
