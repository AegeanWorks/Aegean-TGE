# INVENTED: one band per worker in the kq8 prefill kernel — KILLED (0.93-0.99x; activation re-sweep is not the bottleneck)

- **Idea**: The 2026-11-11 llama.cpp comparison leaves one gap: prefill, 1.3-1.5x behind llama on matched f16 KV. The kq8 prefill kernel bands rows with `band_rows = rows / (threads * 4)`, and inside a band the token loop re-reads the whole Q8_K activation matrix for every weight row. At 512 tokens the Q8_K buffer is ~1.8 MB, so 64 bands re-sweep ~115 MB of L3 per matvec; one band per worker (10 bands) would cut that to ~18 MB and, if activation traffic is the limiter, gain 1.2-1.4x.
- **Cycle date**: 2026-11-11
- **Verdict**: **KILLED.** One band per worker is **0.878/0.985x** at 512 tokens and **0.987/0.939x** at 2,050 (median ~0.95x overall). The activation re-sweep is *not* the bottleneck; reducing band count only loses scheduling granularity.
  Probe that motivated the idea: the kq8 kernel scales 51.4 -> 256.6 -> 347.3 GFLOPS at 1 -> 8 -> 64 tokens (float: 40.8 -> 181.4 -> 278.4), i.e. it is compute-bound at large token counts, not traffic-bound. The llama gap must come from the maddubs chain's issue efficiency/tiling, not from L3 sweep volume.
- **Tree state at close**: spike reverted from `src/cpu/avx2/dequant_batch_kq8_avx2.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

`gemv_dequant_batch_kq8_avx2` (`src/cpu/avx2/dequant_batch_kq8_avx2.cpp:324-387`): quantise each token row to Q8_K once per call, then `pool.parallel_for` over row bands; inside a band, for each weight row, loop tokens and run `dot_q4k_q8k`/`dot_q6k_q8k`. Each band touches the entire Q8_K buffer once per row it owns, so total activation traffic scales with `bands x row_bytes x tokens`.

Spike: `band_divisor = args.tokens > 16 ? threads : threads * 4` (one band per worker for sustained prefill, four for short token counts). Bit-identical output: only the row-to-worker assignment changes and each row's dot is unchanged.

## 2. Predicted gain (with the math)

At 512 tokens x 3,072 rows: 64 bands, each band's rows re-read 512 x 12 blocks x 292 B = 1.8 MB -> ~115 MB per matvec from L3; 10 bands -> ~18 MB. If L3 bytes were the limiter at ~100-150 GB/s, the saving is ~0.6-0.8 ms per matvec; with ~196 matvecs, ~120-160 ms of the ~8.5 s prefill (1.5-2%). If the limiter were L3 *latency* rather than bytes, the gain could be larger. Neither materialised.

## 3. Falsifier (stated before the spike)

**PROCEED** if prefill at 510 or 2,050 tokens gains >= 10% over the four-bands-per-worker binary in alternating pairs, ids identical. **KILL** otherwise.

Measured 0.939-0.987 (median ~0.95). Kill.

## 4. File anchors (current HEAD, read-before-write verified)

- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:324-387` — batch kernel (banding at `:359-363`, token loop at `:380-387`).
- `:143-216` / `:222-313` — `dot_q4k_q8k` / `dot_q6k_q8k` (the compute chain).
- `apps/cli/main.cpp` bench batch path (`command_batch_bench`) — the probe used it with `--tokens 1|8|64 --format q4k --kernel float|kq8`.
- `reports/llama-cpp-comparison-2026-11-11.md` — the gap this cycle tried to close.
- `reports/idea-invented-dot8acc-killed.md` — register-pressure precedent for future tiling work.

## 5. What was done (temporary spike, fully reverted)

Changed the band divisor for `tokens > 16` from `threads * 4` to `threads` with an explanatory comment; built; 171/171; built a four-band baseline from the stashed tree; ran alternating prefill pairs at 510 and 2,050 tokens (`--n 4`, threads 10, f32 KV); reverted; rebuilt; 171/171; gates 4/4; clean tree.

## 6. Measured numbers

`Llama-3.2-3B-Instruct-Q4_K_M.gguf`, `generate --prompt <fox x51|x205> --n 4 --threads 10 --gpu off`.

| tokens | pair | four bands t/s | one band t/s | ratio |
|---:|---:|---:|---:|---:|
| 510 | 1 | 68.0 | 59.7 | 0.878 |
| 510 | 2 | 59.6 | 58.7 | 0.985 |
| 2050 | 1 | 54.8 | 54.1 | 0.987 |
| 2050 | 2 | 54.2 | 50.9 | 0.939 |

Ids unchanged (default path, same arithmetic).

### Probe (bench batch, 3072x3072 Q4_K, best of 3)

| tokens | float GFLOPS | kq8 GFLOPS | kq8/float |
|---:|---:|---:|---:|
| 1 | 40.79 | 51.44 | 1.26x |
| 8 | 181.40 | 256.62 | 1.41x |
| 64 | 278.37 | 347.25 | 1.25x |

The compute-bound character at 64 tokens (347 GFLOPS, still below llama's effective ~450) is where the remaining prefill gap lives, not in activation traffic.

## 7. Why it failed and what it leaves

- Band count is a *scheduling* knob, not a traffic knob on this box: the Q8_K buffer is small enough that the per-band re-read is L3-resident and cheap, and the kernel is limited by the maddubs/madd chain. Cutting bands starves the work-stealing balance for nothing.
- The next real prefill lever is a tiled integer GEMM that reuses the activation block across several *weight rows* inside registers (rows x tokens blocking), raising maddubs issue efficiency toward llama's. That is a new kernel, not a loop-constant change, and any such attempt must respect the 16-YMM register budget (the dot8acc kill): tile at 2 rows, not 4, and measure at `bench batch` before touching the model path.
- Nothing ships; the band divisor stays `threads * 4`.

## 8. Reimplementation recipe (rebuild only to re-verify the kill)

1. In `gemv_dequant_batch_kq8_avx2`, replace `int band_rows = rows / (threads * 4);` with `const int band_divisor = args.tokens > 16 ? threads : threads * 4; int band_rows = rows / band_divisor;`.
2. Build + `tge_tests` 171/171; build a baseline binary from the reverted tree.
3. Alternate `generate --prompt <fox x51|x205> --n 4 --threads 10 --gpu off` pairs and compare prefill t/s -> expect 0.94-0.99x. Revert.
4. For the follow-up GEMM idea: start from `bench batch --rows 3072 --cols 3072 --tokens 64 --format q4k --kernel kq8` (347 GFLOPS baseline), add a two-row dot that loads the Q8_K block once into registers and computes two row dots, and require >= 1.15x at tokens 64 before any model wiring.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge bench batch --rows 3072 --cols 3072 --tokens {1|8|64} --repeat 3 --format q4k --kernel {float|kq8}
  -> 51.4/256.6/347.3 GFLOPS (kq8), 40.8/181.4/278.4 (float)
tge generate ... --prompt <fox x51|x205> --n 4 --threads 10 --gpu off -> §6 A/B table
git checkout -- src/cpu/avx2/dequant_batch_kq8_avx2.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in one throwaway binary (`build/idea/tge-band4-base.exe` is the baseline); the committed tree is unchanged except this report.
