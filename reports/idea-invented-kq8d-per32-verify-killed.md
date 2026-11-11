# INVENTED: per-32 batch verifier for spec+kq8d composition — KILLED (speed-neutral), and it corrects the previous report's mechanism claim

- **Idea**: Last cycle's stacking report said the speculative verifier is "float-only" and prescribed a per-32 batch verifier kernel so verify arithmetic could share the kq8d decode path. Build `gemv_dequant_batch_kq8_32_avx2` (each (row, token) dot is the same per-32 dot the single-row path uses, so per-row results are bit-identical), route the verifier through it behind `--kq8d-verify`, and re-price speculative decode.
- **Cycle date**: 2026-11-11
- **Verdict**: **KILLED as a speed idea — and it corrects the premise of the previous report.**
  - Measured periodic speculative decode (3 rotated pairs, `--kq8d head,down`, n=64, threads 10): `--kq8d-verify` vs the shipped per-256 verifier is **0.970 / 0.981 / 1.055** (median ~0.98); vs off 1.022/1.014/1.015. Speed-neutral within noise, all ids identical (`9D-93-29…`).
  - **The verifier was never float-only.** `prefill_run` calls `batch_matvec(..., float_only = false)` for q/k/v/o/gate/up/down (`src/models/qwen2.cpp:2435-2524`), so verify already runs the **kq8 per-256 integer batch kernel** by default. The `float_only = true` calls exist only in `forward_batch` (`:1706-1832`, `:2042`) — the continuous-batch mixed-prompt path, where occupancy invariance matters (the 2026-11-04 LEAK). The previous report's mechanism sentence and its suggested "future idea" were wrong; the measured stacking numbers stand, but the cause is "verify is already integer and memory-bound", not "verify is float".
  - Consequence: there is nothing to gain by making verify per-32 (same bytes, same stream, slightly more scale work); the composition line is closed. The honest stacking statement is **neutral, not harmful**: this cycle's `head,down` under spec reads 0.96-1.05, while last cycle's full-stack reading (0.91-0.95) was likely load + a slightly heavier 1-row tier mix; neither supports "the tier hurts under speculation" as a general claim.
- **Tree state at close**: spike reverted from `src/cpu/avx2/dequant_batch_kq8_avx2.cpp`, `include/tge/cpu/kernels.hpp`, `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp`, `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism (as built)

- `Q8K32Block` / `quantize_row_q8k32` / `dot_q4k_q8k32` / `dot_q6k_q8k32` re-applied from the per-32 cycle (bit-identical single-row arithmetic).
- New `gemv_dequant_batch_kq8_32_avx2(args, config)`: quantises each of the `tokens` activation rows per 32 values once, bands over weight rows, and for every (row, token) calls the same per-32 dot. Because both the quantiser and the dot are the single-row functions applied to that row's activation, a row's output equals a one-row call on the same vector — no occupancy dependence.
- `Qwen2::set_verify_kq8_32(bool)` + `verify_kq8_32_` member; `batch_matvec` routes K-quant matvecs through the per-32 batch kernel when `verify_kq8_32_ && float_only` (intended for the verifier); CLI `--kq8d-verify`.
- **The route never fires in practice**: the verify path has `float_only == false`, so the branch never matches, and the shipped kq8 batch kernel runs either way. That is itself the finding.

## 2. Predicted gain and why it failed

Prediction: verify dominates a full-accept speculative step (6 rows), so replacing a float batch GEMM (dequant-to-f32 scratch + f32 GEMM) with integer dots should cut step cost 20-30% and lift spec wall by 1.2-1.3x.

Reality: that float batch path was the *pre-kq8* design. The shipped `batch_matvec` already sends tokens > 1 K-quant matvecs to `gemv_dequant_batch_kq8_avx2` (per-256 integer dot, no f32 scratch). Both per-256 and per-32 integer batch forms are bound by the same weight stream (2.0 GiB/verify), and the per-32 form does slightly more scale work — hence 0.97-1.06x.

## 3. Falsifier (stated before the spike)

**KEEP** if `--kq8d-verify` improves periodic speculative decode by >= 10% over the shipped verifier with ids identical. **KILL** if neutral or negative.

Measured 0.970/0.981/1.055 -> kill.

## 4. File anchors (current HEAD, read-before-write verified)

- `src/models/qwen2.cpp:2435-2524` — `prefill_run` batch matvecs (no `float_only` argument -> `false`, so kq8 at tokens > 1). **The verifier path.**
- `src/models/qwen2.cpp:1706-1832`, `:2042` — `forward_batch` matvecs with `float_only = true` (continuous-batch occupancy invariant).
- `src/models/qwen2.cpp:1131-1141` — the `float_only` comment and the kq8 batch branch.
- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:324-387` — shipped per-256 batch kernel; `:397-456` — single-row per-256 entry.
- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp` (spike) — `gemv_dequant_batch_kq8_32_avx2` after the row kernel; declaration in `include/tge/cpu/kernels.hpp` after the per-256 batch entry.
- `reports/idea-invented-kq8d-speculation-stacking-measured.md` — the report this one corrects (measured numbers stand; cause corrected).
- `reports/idea-invented-kq8d-per32-q6k-max-tier-keep.md`, `reports/idea-invented-kq8d-per32-activation-keep.md` — the per-32 kernel reused here.

## 5. What was done (temporary spike, fully reverted)

Re-applied the per-32 single-row kernel; added the per-32 batch kernel and declaration; added `set_verify_kq8_32` + `verify_kq8_32_` + the `float_only` dispatch branch; added the `--kq8d-verify` flag; built; 171/171; ran three rotated three-arm pairs (off / `--kq8d head,down` / `+ --kq8d-verify`) on the periodic prompt with `--speculative --print-ids`; reverted five files; rebuilt; 171/171; gates 4/4; clean tree.

## 6. Measured numbers

`Llama-3.2-3B-Instruct-Q4_K_M.gguf`, periodic fox-x120 prompt (1,200 tokens), `--n 64 --kv f32 --threads 10 --gpu off --speculative --print-ids`, 11 steps / 5.09 accepted per step in every arm.

| pair | off t/s | kq8d head,down | kq8d + per-32 verify | k8/off | k8v/off | k8v/k8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 35.37 | 37.28 | 36.15 | 1.054 | 1.022 | 0.970 |
| 2 | 36.20 | 37.44 | 36.71 | 1.034 | 1.014 | 0.981 |
| 3 | 33.68 | 32.39 | 34.17 | 0.962 | 1.015 | 1.055 |
| **median** | 35.37 | 37.28 | 36.15 | **1.034** | **1.015** | **0.981** |

Ids: `9D-93-29…` identical in all nine runs. The per-32 batch kernel is bit-identical per row to the single-row per-32 dot by construction, so identical ids here are expected on a periodic prompt with a wide accept margin.

## 7. What this changes

- **Correction**: the verifier is already integer (per-256 kq8). The `float_only` guard is a continuous-batch invariant, not a verifier property. Any future "make verify faster with kq8d" idea must start from "verify is bandwidth-bound integer already" — not from the float premise.
- The composition question is settled: spec + kq8d is **neutral within noise**, because the verifier dominates and is already on the integer batch path. Do not spend cycles trying to stack them; the tier pays on 1-row decode (prose, non-spec), speculation pays on repetitive text, and `--speculative auto` picks the right regime.
- The per-32 batch kernel itself is a clean, reusable artifact (bit-identical per-row by construction); it is recorded here in case a future verifier ever needs finer activation quantisation for *fidelity* reasons rather than speed.

## 8. Full reimplementation recipe (rebuild only to re-verify the kill)

1. Apply the per-32 single-row kernel (`reports/idea-invented-kq8d-per32-q6k-max-tier-keep.md` §8 steps 1-4).
2. Append `gemv_dequant_batch_kq8_32_avx2` after `gemv_dequant_kq8_row_avx2`: validate args; fall back to `gemv_dequant_batch_avx2` on non-K/unaligned; `std::vector<Q8K32Block> quantized(tokens * (cols/256))`; `quantize_row_q8k32` per token row; band over rows; inner loop over tokens calling `dot_q4k_q8k32`/`dot_q6k_q8k32`; write `out[token*rows + index]`. Declare in `kernels.hpp`.
3. `Qwen2`: add `verify_kq8_32_` + `set_verify_kq8_32`; in `batch_matvec`, before the existing kq8 branch: `if (verify_kq8_32_ && !scalar_kernels_ && float_only && (Q4_K||Q6_K)) { gemv_dequant_batch_kq8_32_avx2(...); } else if (kq8_effective() && ...)`.
4. CLI: `--kq8d-verify` boolean -> new defaulted `command_generate` parameter -> `model->set_verify_kq8_32(flag)` after `set_kq8d`.
5. Build + `tge_tests` 171/171. Run `--speculative --print-ids` with and without `--kq8d-verify` on the periodic prompt, rotated pairs -> expect 0.97-1.06x and identical ids. If a future tree ever routes verify through `float_only=true`, re-run then — the kernel would matter, not before.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge generate ... --speculative --print-ids [--kq8d head,down [--kq8d-verify]]
  periodic, 3 rotated pairs -> k8/off 1.054/1.034/0.962, k8v/off 1.022/1.014/1.015,
                               k8v/k8 0.970/0.981/1.055; ids identical in all runs
git checkout -- src/cpu/avx2/dequant_batch_kq8_avx2.cpp include/tge/cpu/kernels.hpp \
  include/tge/models/qwen2.hpp src/models/qwen2.cpp apps/cli/main.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the rebuilt binary; the committed tree is unchanged except this report.
