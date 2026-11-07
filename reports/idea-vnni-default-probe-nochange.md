# IDEA-XXXX: VNNI Default-Flip Probe — NO CHANGE (inconsistent sign, stays opt-in)

- **Idea**: Route Q8 through the VNNI kernel by default (it exists, is opt-in, and llama.cpp uses int8 always) to close the llama.cpp gap (0.5B: −31% prefill / −19% decode vs b11960).
- **Cycle date**: 2026-11-07
- **Verdict**: NO CHANGE. No code touched (`git status` clean throughout); the opt-in flag stays opt-in. Four interleaved default-vs-`--vnni` runs on 0.5B-Q8_0 (128 prompt + 64 decode, 6 threads) flip sign on both phases, so there is no shippable win on this box. Nothing pushed.
- **Tree state at close**: clean, no diff, stash untouched (pre-existing entry only). `master` ahead of `origin/master` by 155 after this report.

## 1. Mechanism tested (no implementation — existing flag)

`--vnni` routes Q8 GEMV/GEMM through `gemv_q8_batch_vnni` (activation quant once per call, maddubs dots) instead of the float dequant path. Question: does it win consistently enough to become the default?

## 2. Measured numbers (0.5B-Q8_0, 128+64, 6 threads, interleaved)

| run | default prefill | vnni prefill | default decode | vnni decode |
|---|---|---|---|---|
| A | 256.0 t/s | 294.1 (+14.9%) | 63.63 t/s | 67.99 (+6.9%) |
| 1 | 312.3 | 312.7 (+0.1%) | 77.64 | 75.15 (−3.2%) |
| 2 | 295.0 | 306.9 (+4.0%) | 72.93 | 65.19 (−10.6%) |
| 3 | 300.2 | 291.2 (−3.0%) | 67.09 | 70.39 (+4.9%) |

Ids byte-identical default-vs-vnni on the paired run. Fidelity `--vnni` (arch corpus, 1023 pos): KL 1.485e-03 / worst 2.302e-02 / top-1 994-1023 — inside the priced VNNI tier, but 29 flips.

## 3. Verdict and why

Both phases flip sign across runs (prefill +15/+0/+4/−3, decode +7/−3/−11/+5) — smaller than box variance, no consistent direction. A default flip that sometimes regresses decode 11% is not shippable, regardless of the single +15% run. The 29-flip fidelity is priced but not free. The flag stays opt-in for kernels where it is load-proven per call site; revisit only with a quiet-box best-of-5 showing same-direction ≥10% on both phases.

## 4. Exact commands run and their results

- `tge generate --model 0.5B-Q8_0 --tokens "<128 ids>" --n 64 --threads 6 --gpu off [--vnni] [--print-ids]` ×8 → §2 table, ids identical paired run.
- `tge fidelity --model 0.5B-Q8_0 --corpus docs/architecture.md --vnni` → KL 1.485e-03, 994/1023.
- No build (no code change); `git status` clean; report committed; no push.

(End of file.)
