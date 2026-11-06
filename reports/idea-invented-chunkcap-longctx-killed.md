# INVENTED: GQA Chunk-Count Sweep at 8k KV (L2-Cap Follow-up) — KILLED (plateau holds; +3.5% best, under the bar)

- **Idea**: Invented follow-up to the chunk-cap probe (which killed L2-cap chunking at 3.6k KV): re-test at ~8k-token KV where attention is 38% of decode (vs 20%), giving chunk sizing its best possible case — N ∈ {8, 16, shipped ~10} plus the short-context N=1 anchor.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED — N=16 beats the shipped formula by 3.5% best-of-3 (59.29 vs 57.30 t/s, no run overlap), under the pre-registered 5% bar; N=8 ties baseline, N=1 halves throughput. Diminishing returns set in immediately past the default: parallelism saturates, cache was never the constraint (working sets stay L1/L2-resident even at 8k). No L2-cap policy is owed at any measurable span. Spike fully reverted (byte-identical diff verified); only this report committed.
- **Tree state at close**: `git status` clean apart from PRE-EXISTING, UNCOMMITTED peer spike (verified byte-identical to the saved baseline before AND after — untouched). No spike branch remains. `master` ahead of `origin/master` by 49 after this report (reports only). Nothing pushed (human gate).

## 1. Mechanism

Env-gated `TGE_GQA_CHUNKS` override of the GQA decode chunk count, clamped to
[1, kMaxGqaChunks] (group scratch is sized for 16 — confirmed hard ceiling,
unchanged from the prior probe). Same binary A/B via env (unset = shipped
formula), so no binary variance contaminates the comparison.

## 2. Predicted gain — none claimed (diagnostic)

Falsifier (stated): kill unless some chunk count beats the shipped formula
by >5% best-of-3 decode t/s at ~8k KV with identical ids. The bar is the
same as the short-context probe (comparability across the two cycles).

## 3. Falsifier — trips (best-of-best +3.5% < 5%)

N=16 consistent direction (3/3 above baseline best-of-3 range) but under the
bar; N=8 ties; N=1 collapses (−50%: 2 tasks on 10 threads). The effect is
real, small, and saturating — exactly a parallelism knob, not a cache knob.

## 4. File anchors (current `master`, untouched except by the reverted spike)

- `src/models/qwen2.cpp` (`attention_gqa` chunk formula, `kMaxGqaChunks`
  scratch bound) — the only two hunks the spike touched.
- `reports/idea-invented-chunkcap-killed.md` (short-context probe this
  extends; long-context was its explicit open question).

## 5. What was done (temporary spike, fully reverted with verification)

1. `git checkout -b spike-chunkcap-long` (no commit ever made on it).
2. Re-applied the override (2 hunks, `qwen2.cpp` only).
3. Build clean; `ctest` pass; 141/141.
4. 2081-token prompt + `--n 6000` + `--context 8192` → ~8k-token KV,
   0.5B-Q8_0, 10 threads, `--gpu off`: baseline best-of-3, N=16 best-of-3,
   N=8 ×2, N=1 ×1 (short-context anchor from the prior probe: 30 t/s).
5. Thinking set: fox ids identical N=16 vs unset (0.5B); Llama-3B haiku
   matches the recorded baseline ids.
6. Inverse-edited both hunks; `git diff` byte-identical to the saved peer
   baseline (no normalization needed this time).
7. Deleted branch; rebuilt pristine; re-ran full suite green.
8. `aikit.py check` 2/4 (test + secrets pass; hygiene + guidelines fail
   pre-existing).

## 6. Measured numbers (0.5B-Q8_0, ~8k-token KV, 10T, decode t/s best-of-3)

| chunks | best-of-3 | best | vs shipped best (57.30) |
|---|---|---|---|
| shipped (~10) | 57.30 / 55.55 / 56.14 | 57.30 | — |
| N=16 | 57.63 / 59.29 / 59.16 | 59.29 | **+3.5%** |
| N=8 | 55.16 / 56.19 / (2 runs) | 56.19 | −1.9% |
| N=1 | (short-ctx anchor) ~30 | — | −50% |

Attention share at this length: 38% of decode (vs 20% at 2k) — the regime
where chunking matters most, and it still does not clear the bar.

## 7. Full reimplementation recipe (kill-grade — rerun only to re-verify)

1. `qwen2.cpp`: `int chunks = max(1,(threads*2+kv_heads-1)/kv_heads)`,
   clamp to `kMaxGqaChunks`; env `TGE_GQA_CHUNKS` override in [1,16];
   `#include <cstdlib>`.
2. Gate: 2081-token prompt + `--n 6000 --context 8192`, threads 10,
   `--gpu off`, best-of-3 per setting; expect §6 (N=16 +3–4%, N=8 tie,
   N=1 collapse); ids must match everywhere.
3. Revert both hunks; verify `git diff` byte-identical to pre-spike tree.

## 8. Verdict and why

**KILLED — and with it the whole L2-cap line of inquiry at every span this
box can measure.** Two independent sweeps (3.6k and 8k KV) agree: chunk
count is a parallelism knob already saturated by the shipped formula.
N=16's consistent-but-sub-bar +3.5% is the signature of merge-overhead
tradeoffs near saturation, not of a cache effect waiting for a cap policy.
Do not revisit without (a) spans where a 16-chunk working set exceeds L3
(tens of k tokens, beyond the build's context/test harness), or (b) a
scratch resize enabling >16 chunks with measured (not hoped) headroom.

## 9. Exact commands run and their results

- `git checkout -b spike-chunkcap-long` → new branch.
- 2-hunk spike → build 54/54 → 141/141.
- Long-decode A/B: baseline 57.30/55.55/56.14; N=16 57.63/59.29/59.16;
  N=8 55.16/56.19; N=1 short-ctx anchor ~30.
- Thinking-set ids: identical everywhere (incl. recorded Llama ids).
- Inverse edit ×2 → diff byte-identical to baseline → delete branch →
  rebuild → 141/141 → `aikit.py check` 2/4.
- No push.
