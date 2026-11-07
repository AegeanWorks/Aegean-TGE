# IDEA-1582 FOLLOW-UP: Decode Integer Path, Round-2 — KILLED again (1013/1023, deterministic max|diff| 0.66)

- **Idea**: Reopen §63 ("Decode integer path ... reverted") via its own clause: a differential on outlier-bearing inputs with a priced gate. If the kq8d row kernel tracks float on outliers AND model fidelity passes ≤1.6e-3 with top-1 held, wire decode-int as default; else revert.
- **Cycle date**: 2026-11-07
- **Verdict**: KILLED (second time). Reverted completely: `git status` clean, no diff, stash untouched, spike branch never created (throwaway edits on master, all reverted), rebuilt master 54/54, tests 141/141. Nothing pushed (human gate).
- **Tree state at close**: clean, `master` ahead of `origin/master` by 154 after this report.

## 1. Mechanism (as spiked, reverted)

Temp-only, all reverted: (a) `Qwen2::matvec` Q4_K/Q6_K branch behind `TGE_KQ8D=1` calling `gemv_dequant_kq8_row_avx2` (default keeps the float path); (b) temp `spike-kq8d-outlier` driver comparing shipped float AVX2 vs the row kernel on random Q4_K bytes with uniform and outlier (tiny + one 100x channel) activations at cols 1024/3072/8192.

## 2. Predicted gain — not reached (fidelity gate failed first)

Bench pulse on record (2–2.8x kernel) was not re-pulsed: the fidelity arm decides, and it failed, so no end-to-end A/B was owed. Falsifier: kill if fidelity KL > 1.6e-3 OR top-1 agreement breaks vs the 1023/1023 float control.

## 3. Falsifier outcome — kill on top-1

Outlier differential (new evidence): kernel tracks float with rel ≤ 0.006 on ALL shapes, outlier mode CLEANER than uniform (0.0013–0.0063 vs 0.005–0.0062) — the leading benign hypothesis (per-256-block amax crushing from §63 candidate (a)) does NOT reproduce on synthetic outliers. So the kernel is exonerated at the unit level.

Model fidelity, 0.5B-Q4_K_M, arch corpus, 1023 positions, interleaved best-of-2 (deterministic — both runs digit-identical):

| arm | KL mean | KL worst | max\|diff\| | top-1 |
|---|---|---|---|---|
| float decode (control) | 1.186e-11 | 1.933e-09 | — | 1023/1023 |
| + decode-int (TGE_KQ8D=1) | 7.350e-04 | 2.197e-02 | 6.581e-01 | **1013/1023** |

KL passes the 1.6e-3 VNNI-class bar, but 10 argmax flips with max|diff| 0.66 is a real quality change (baseline: zero flips) — worse than baseline → revert per protocol. (Note the signature differs from §63's 4.5e-3/2-flips: different corpus/positions, same conclusion.)

## 4. File anchors (all reverted)

- `src/models/qwen2.cpp` (`Qwen2::matvec` Q4_K/Q6_K branch + `<cstdlib>` — reverted).
- `apps/cli/main.cpp` (temp `command_spike_kq8d_outlier` + dispatch — reverted).
- Grounds: `docs/measurement-log.md` §63 (the revert being re-tested), round-5 entry gates in §52 (the reopen clause this round executes).

## 5. Why killed, and what would reopen it now

The unit kernel is innocent on synthetic outliers, yet real-model max|diff| hits 0.66 deterministically. Remaining candidates: (a) real activation structure (many moderately-large channels, not one 100x spike) loses precision coherently under per-256-block scales — needs a differential driven by CAPTURED real activations, not synthetic; (b) a transcription slip invisible to random/zero/constant/outlier-synthetic inputs. What reopens: captured-real-activation differential with a priced gate, or proof of what llama.cpp runs for single-token decode (e.g., per-32-block activation scales, which would confirm (a) and prescribe finer granularity). Do NOT re-run synthetic-input differentials — this round exhausted them.

## 6. Exact commands run and their results

- Baseline binary saved to Temp/kilo/kq8d-port (pre-change master); `cmake --build --preset release` → 54/54; `tge_tests` → 141/141.
- Temp wiring + driver; rebuild 54/54; suite 141/141.
- `tge spike-kq8d-outlier` → 6/6 rel ≤ 0.0063, exit 0.
- `TGE_KQ8D=1 tge fidelity --model 0.5B-Q4_K_M --corpus docs/architecture.md` ×2 → KL 7.350e-04 / worst 2.197e-02 / maxdiff 0.658 / 1013-1023, digit-identical across runs.
- Control (env unset) → floor 1.186e-11 / 1023-1023.
- Reverted both files (empty diff verified), rebuilt master 54/54, `tge_tests` 141/141, `git status` clean, stash untouched; report committed; no push.

(End of file.)
