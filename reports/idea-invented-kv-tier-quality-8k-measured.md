# MEASURED: long-context KV tier quality at 8k — f16 holds (99.98%), i8 fails its own bar (96.60%), sparse collapses (67.49%)

- **Idea**: The sparse-8k correction withdrew the "sparse is a general long-context lever" guidance and pointed at the dense/f16/i8 KV tiers as the quality-preserving alternatives — but that recommendation was itself unpriced at 8k. i8's bar (KL <= 1e-3, top-1 >= 98.6%) was established at 2,048 positions (1997/1999, 99.9%); the tier's error could compound through the KV state at 8k just as sparse's did. The harness compares a full-scalar f32-KV reference against the candidate tier, so one run per tier prices it at the target context. Run f16 and i8 at 8,199 positions on the same corpus as the correction.
- **Cycle date**: 2026-11-13
- **Verdict**: **MEASURED — only f16 survives at 8k; i8 drops below its own bar and sparse is far outside it.** Qwen2.5-0.5B-Instruct-Q8_0, `architecture.md` x 9 corpus, `--context 8704 --tokens 8200 --threads 10 --gpu off`, 8,199 teacher-forced positions, candidate vs full-scalar f32-KV reference:
  - **f16 KV**: KL **1.145e-06**, worst 5.606e-04, max |diff| 9.829e-02, top-1 **8197/8199 (99.98%)** — passes everywhere, and the 8k number is slightly better than the 497-position precedent (1.7e-6).
  - **i8 KV**: KL **1.657e-02**, worst 8.517e+00, max |diff| 1.183e+01, top-1 **7920/8199 (96.60%)** — **2.0pp below the 98.6% dossier bar** and ~12.7x the 2k mean KL (1.303e-3). The tier does not hold at 8k.
  - **sparse (for the ladder, from the correction run)**: default 512/4 **67.49%** (KL 1.630), stride 16 53.13%, stride 2 75.82%.
  - **Ladder at 8k**: f16 (-0.02pp) >> i8 (-3.40pp, bar fail) >> sparse default (-32.49pp). The 8k quality-preserving choice is f16; i8 is a capacity mode with a measured quality cost; sparse is a deliberately lossy speed mode.
  - **Speed context unchanged**: i8 at 8k decode was already measured as a loss (-13 to -23%, `idea-invented-i8-8k-killed.md`); f16 is the tier that both halves and the KV and holds quality; sparse is the only large speed lever and is the most lossy.
- **Tree state at close**: no code touched (shipped flags only); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Qwen2.5-0.5B-Instruct-Q8_0 |
| corpus | `docs/architecture.md` repeated 9x (107,624 bytes) |
| positions | 8,199 usable, `--context 8704 --tokens 8200` |
| threads / GPU | 10 / off |
| harness | candidate KV tier + production kernels vs full-scalar f32-KV oracle (`tge fidelity`) |
| arms | f16, i8 (this cycle); sparse 512/4, 512/2, 512/16 (correction run, same corpus) |

The reference model in `run_fidelity` is loaded with f32 KV and `set_scalar_kernels(true)` + Scalar attention; the candidate is loaded with the requested tier and production kernels (`apps/cli/main.cpp:2520-2540`). The printed KL therefore prices both the tier's storage rounding and the vectorization against the exact double-precision-path oracle, and cross-arm differences at the same context are the tier's price.

## 2. Results

| arm | mean KL | worst KL | max |diff| | top-1 | vs f32 oracle | vs i8 2k bar |
|---|---:|---:|---:|---:|---:|---:|
| f16 (8,199 pos) | **1.145e-06** | 5.606e-04 | 9.829e-02 | **8197/8199 (99.98%)** | **-0.02pp** | pass |
| i8 (8,199 pos) | 1.657e-02 | 8.517e+00 | 1.183e+01 | **7920/8199 (96.60%)** | **-3.40pp** | **FAIL (-2.00pp)** |
| sparse 512/2 (~53% kept) | 1.250e+00 | 1.940e+01 | 2.710e+01 | 6216 (75.82%) | -24.16pp | fail |
| sparse 512/4 (~30% kept) | 1.630e+00 | 1.833e+01 | 2.858e+01 | 5533 (67.49%) | -32.49pp | fail |
| sparse 512/16 (~12% kept) | 2.382e+00 | 1.825e+01 | 3.005e+01 | 4356 (53.13%) | -46.85pp | fail |

### i8 across context (same corpus family, same model)

| context | positions | mean KL | top-1 | source |
|---|---:|---:|---:|---|
| 2,048 | 1,999 | 1.303e-03 | 1997/1999 (99.90%) | i8 long-context confirm (2026-11-06) |
| 8,199 | 8,199 | **1.657e-02** | **7920/8199 (96.60%)** | this cycle |

i8's mean KL grows 12.7x for 4.1x the positions — slightly superlinear, and the top-1 rate falls 3.3pp. The tier's 2k flatness did not transfer.

## 3. Reading

- **f16 is the only 8k tier that is effectively free.** 99.98% top-1 and KL 1.15e-6 at 8k; the two flips are near-ties of the same class as the 497-position precedent. f16 halves the KV versus f32 and is the recommended long-context tier.
- **i8's error compounds.** At 2k it sat just above the 98.6% bar (99.9%); at 8k it sits below it (96.6%). The mechanism matches the sparse result: quantization error at every layer feeds the next position through the cache, so a tier that is near-exact at short context accumulates at long context. i8 remains useful as a *capacity* mode (KV quartered vs f32) for workloads that tolerate a ~3.4pp top-1 cost, and its 8k decode speed is a measured loss anyway, so it is not a performance mode.
- **The sparse ladder now reads correctly**: at 8k the KV-tier lever (f16) is free, i8 costs 3.4pp, and sparse costs 24-47pp. Any 8k recommendation that starts from sparse is a lossy-speed recommendation, not a quality one.
- **Consistency with the speed reports**: the 8k speed measurements (sparse decode 1.36-1.77x, spec+sparse 1.71-1.83x on code, block-sparse prefill 1.18-1.27x) stand as speed claims on their workloads. Their quality price is now fully priced at 8k, and the "robust 8k lever" wording from the reopening report is superseded by the correction + this tier ladder.
- **What to quote for 8k today**: f16 KV, dense attention, optional speculation on echo-heavy prompts. i8 only when memory forces it and 96.6% top-1 is acceptable. Sparse only for deliberately lossy speed work, with the 24-47pp price stated.

## 4. Falsifier (stated for this diagnostic)

**KEEP the tier ladder** if f16 at 8,199 positions stays within 0.5pp of the oracle and i8 stays within its 98.6% bar. **CORRECT** if either fails. Measured f16 -0.02pp (pass) and i8 -3.40pp (fail, 2.0pp below the bar): the ladder is corrected — f16 is the 8k tier; i8 is a capacity trade, not a quality-preserving tier.

## 5. File anchors (current HEAD, read-before-write verified)

- `apps/cli/main.cpp:2520-2545` — fidelity reference (f32 KV, scalar kernels, Scalar attention) vs candidate (requested tier, production kernels); this is what makes the KL a tier price.
- `include/tge/models/qwen2.hpp` — KV tier members and `Qwen2::load(..., kv_half, kv_q8)`.
- `src/models/qwen2.cpp` — `store_kv`/attention tier reads (f16 widen path, i8 dequant path).
- `apps/cli/main.cpp:2740+` — `--kv` parsing in fidelity.
- `reports/idea-invented-sparse-8k-quality-correction.md` — the sparse ladder rows and the withdrawal this cycle completes.
- `reports/idea-invented-i8-longctx-confirmed.md` — the 2k i8 bar (1.303e-3, 1997/1999).
- `reports/idea-invented-i8-8k-killed.md` — the 8k i8 speed loss (-13 to -23%).

## 6. Reimplementation recipe

1. Corpus: `docs/architecture.md` x 9 -> 107,624 bytes; confirm with `--context 64 --tokens 10` that the corpus reports 8,199 positions at ctx 8704.
2. Run each tier (each ~8-11 minutes):

   ```text
   tge fidelity --model qwen2.5-0.5b-instruct-q8_0.gguf --corpus <corpus8k> \
     --context 8704 --tokens 8200 --kv {f16|i8} --threads 10 --gpu off
   ```

   Expect f16 KL 1.145e-06 / 8197/8199 and i8 KL 1.657e-02 / 7920/8199. The dense f32 arm is the harness reference by construction, so no separate run is needed.
3. Do not extrapolate the 2k tier numbers to 8k; both i8 and sparse degrade materially.
4. No code ships.

## 7. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56, cycle gate)
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed

fidelity (0.5B Q8_0, ctx 8704, 8200 tokens, 10 threads, gpu off):
  --kv f16 -> KL 1.145e-06, worst 5.606e-04, max|diff| 9.829e-02, 8197/8199, 507 s
  --kv i8  -> KL 1.657e-02, worst 8.517e+00, max|diff| 1.183e+01, 7920/8199, 495 s
git status --short                                -> clean (no code touched)
python aikit.py check aegean-tge                  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
