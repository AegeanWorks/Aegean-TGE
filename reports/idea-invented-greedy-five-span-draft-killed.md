# INVENTED: greedy five-span draft + sixth extension — KILLED (no effect on available prompts)

- **Cycle**: 22
- **Date**: 2026-11-09
- **Family**: speculative decoding / draft selection
- **Verdict**: **KILL.** Behavior-preserving, but the accepted-draft statistics are
  identical on every workload; the t/s deltas are box noise.
- **Tree after the cycle**: spike reverted, branch `idea/greedy-five-span-draft`
  deleted, this report is the only change

## Mechanism

Cycle 18's sampled keep extended the five-token proposal with a sixth token taken
from the most recent earlier copy of the 5-gram, instead of asking
`prompt_lookup_draft` for six. The matcher couples span eligibility to the
requested continuation count (`length >= span + want`,
`length - 1 - end >= want`, and the fallback calls), so `want = 6` can skip a
long span that has only five continuations and choose a shorter span instead.

The greedy path still called `prompt_lookup_draft(history, draft, 6)`. The spike
applied the cycle-18 shape there: call with `need = 5`, then set
`draft[5]` from the most recent earlier copy of `draft[0..4]`, falling back to
`draft[4]`. Under greedy a wrong sixth is harmless for ids: the verifier still
emits the argmax, so the output must stay byte-identical.

Hypothesis: preserving the better five-token span could raise accepted
drafts/step on prompts where the six-continuation requirement changes the match.

## Falsifier

Ids must stay identical (mandatory). Keep if accepted/step or decode tok/s
improves on any workload without a regression; otherwise kill.

## Measured — greedy, `--speculative`, seed 7, 6 threads, alternating pairs

| workload | base best t/s | spike best t/s | pair ratios | ids | accepted/step base -> spike | steps (base/spike) |
|---|---:|---:|---|---:|---:|---:|
| 0.5B Q8_0 story, n=64 | 75.39 | 76.21 | 1.036, 1.034, 1.011, 1.030 | identical | 1.25 -> 1.25 | 4 / 4 |
| 0.5B Q8_0 locked `ha`, n=128 | 172.85 | 182.05 | 1.008, 1.204 | identical | 5.05 -> 5.05 | 21 / 21 |
| 3B Q4_K_M story, n=64 | 34.59 | 35.56 | 1.092, 1.028 | identical | 4.73 -> 4.73 | 11 / 11 |

The accepted-draft counts, accepted/step and verify-step counts are identical in
**every** workload. Only wall-clock differs, within this box's noise (the same
0.5B locked pair swings 1.008/1.204 with an identical verify trace). The
proposal sets were already the same on these histories.

## Why it fails

The span-coupling only changes the selected match when the best long match has
fewer than six continuations. The story and locked prompts are repetitive, so
every eligible span has ample continuations; `need = 6` and five-plus-extension
select the same first five drafts and the same sixth. The fix is
behavior-preserving but unproven on the corpus we have.

This does not contradict cycle 18: that keep used the verifier's already-computed
sixth row (the sampled path drafted five and padded), a different mechanism with
a measured 1.257x on locked text. Here the row was already in use; only the
selection rule changed, and it did not.

## Disposition

KILL for the greedy variant; no integration. Reopen only with a corpus/tail that
contains a rare n-gram whose best match has only five continuations, and judge
it by the `spec` accepted/step, not wall-clock. Otherwise leave the matcher as
is; cycle 18's sixth-draft keep stands on its own.

Tests 167/167; `aikit.py check aegean-tge` 4/4. Nothing was pushed; 29 commits
remain ahead of `origin/master`.
