# CORRECTION: the sampled-spec "53-55% sampler" is a timer double count — the real sampler bill is ~10 ms per run (0.2-0.3% of decode)

- **Idea**: `idea-invented-sampled-spec-8k-measured.md` reported `timers sample 1900-2248 ms (53-55%)` on the speculative decode and concluded the six-row softmax/cut is the long-context sampled-spec bottleneck, naming a batched cut as the next lever. The arithmetic already contradicted it: forward 3594.6 ms + sample 1899.9 ms = 5494.5 ms exceeds the 3613.5 ms decode wall, so the sample window must enclose the verify forward. The earlier `timers sample` on the off path (11.6-13.1 ms) never showed this because the off path has no verify. Spike exact counters around `softmax_probabilities` and `apply_sampling_cuts` only, measure both with the top-p cut and with it disabled, revert.
- **Cycle date**: 2026-11-13
- **Verdict**: **CORRECTED — the sampler is negligible; the prior 53-55% share was the verify forward being counted twice.** Llama-3.2-3B-Instruct-Q4_K_M, 8,193-token periodic prompt, `--context 8704 --n 32 --kv f16 --threads 10 --gpu off --temp 0.8 --seed 7 --speculative`:
  - **top-p 0.9**: decode **8.17 t/s** (7 steps, 3.29 accepted/step, 29 emitted); `timers sample 2022.9 ms (53.3%)`; **real sampler `softmax 2.46 ms over 29 rows, cut 7.55 ms`**.
  - **top-p 1.0 (cut disabled)**: decode **11.06 t/s** (6 steps, 4.33 accepted/step, 31 emitted); `timers sample 1610.1 ms (57.5%)`; **real sampler `softmax 2.14 ms over 31 rows, cut 0.00 ms`**.
  - Real sampler work is **~10 ms per run (0.2-0.3% of the 2.8-3.8 s decode)**, not ~2 s. The `sample` timer in the sampled speculative block starts before `penalized_softmax` and closes after the six-row `verify_draft_logits` and emission, so it re-times the forward that `forward` already counted.
  - **The 1.44-1.72x speed result stands** (this cycle measured 8.17 t/s at top-p 0.9, inside the earlier 7.64-8.58 band). The 26% no-cut difference comes from **acceptance and step count** (4.33 vs 3.29 accepted/step, 6 vs 7 steps), not from the 7.55 ms cut; disabling top-p changes the sampled distribution and is a config choice, not a free win.
  - **Superseded**: "sampler is 53-55% of spec wall", "the growth path is the sampler", and "a batched cut is the named next lever" in `idea-invented-sampled-spec-8k-measured.md` are withdrawn. The spec decode cost is the same verify forward as the greedy path; there is no sampler bottleneck at 8k.
- **Tree state at close**: timer spike reverted from `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism (timer semantics)

`apps/cli/main.cpp:1143-1205` (sampled speculative block): `st0` is taken before `penalized_softmax(logits)`, then the block may call `verify_draft_logits` (timed into `fwd_seconds`), run the per-row `penalized_softmax` loop, emit tokens, and finally `samp_seconds += st1 - st0`. The verify forward therefore lands in **both** counters. On the off path (`:1265-1275`) the sample region is just the draw, which is why its timer reads 11.6-13.1 ms and agrees with the spike's ~10 ms.

The spike added three counters and one print line: `samp_softmax_seconds` around `softmax_probabilities`, `samp_cut_seconds` around `apply_sampling_cuts`, `samp_rows` per row. It ran with the cut (top-p 0.9) and without (top-p 1.0), then was reverted. The softmax cost is ~0.08 ms per row (2.46 ms / 29 rows), the cut ~0.26 ms per row (7.55 ms / 29), both far below the 271-321 ms per step the old timer implied.

## 2. Results

| run | decode t/s | spec line | old `timers sample` | real softmax | real cut |
|---|---:|---|---:|---:|---:|
| top-p 0.9 | **8.17** | 7 steps, 23 acc (3.29/step), 29 emitted | 2022.9 ms (53.3%) | **2.46 ms / 29 rows** | **7.55 ms** |
| top-p 1.0 | **11.06** | 6 steps, 26 acc (4.33/step), 31 emitted | 1610.1 ms (57.5%) | **2.14 ms / 31 rows** | **0.00 ms** |

Per-row softmax: 0.085 ms (top-p 0.9) and 0.069 ms (top-p 1.0). Per-row cut (when active): 0.26 ms. At 29-31 rows per 8k run the entire sampler is ~10 ms, i.e. **0.26-0.36% of the 2.8-3.8 s decode**, and ~0.2% of the spec wall. The pre-spike off-path timer (11.6-13.1 ms for 31 rows, no verify inside) independently matches.

The no-cut arm's faster decode is explained by the spec line, not the sampler: 6 steps at 4.33 accepted/step versus 7 steps at 3.29. Disabling top-p changes which tokens are sampled and therefore how many drafts accept; it is a different distribution, not an optimization.

## 3. Reading and supersession

- **No sampler bottleneck at 8k.** The sampled cost is ~10 ms/run; the speculative decode is the verify forward (matvec + attention), identical in structure to greedy. Sampled and greedy 8k ratios should therefore agree, and they do within noise (greedy 1.42-1.61x, sampled 1.44-1.72x).
- **Withdrawn recommendations**: a batched six-row cut, a fused softmax/cut, and any sampler-side work as the next long-context lever. The measured headroom is 10 ms out of 3,800 ms.
- **The 2k sampled report's sampler interpretation is unverified** (`idea-invented-sampled-spec-2k-measured.md` estimated ~25 ms/token from the same overlapping timers). Its 2.28x speed result stands; its sampler-share attribution should not be quoted without a re-run of this spike at 2k.
- **What the correction does not change**: sampled speculation is still worth ~1.5x at 8k; the KV tier result (f16 fastest and quality-preserving) still implies `--kv f16 --speculative` as the quality-preserving 8k speed config. The only change is where the remaining time actually goes (the verify forward), so any future speculation work must attack the verify, not the sampler.

## 4. Falsifier (stated before the spike)

**CORRECT the timer interpretation** if the exact softmax+cut counters total under 100 ms per run while the old `timers sample` reads over 1 s. **CONFIRM the sampler bottleneck** if the exact counters reproduce the ~2 s figure. Measured 10.0 ms vs 2,023 ms: correct. The off-path timer and the per-row arithmetic agree, so the result does not rest on a single print.

## 5. File anchors (current HEAD, read-before-write verified; spike reverted)

- `apps/cli/main.cpp:1119-1142` — `penalized_softmax` lambda (softmax then cuts), the site instrumented.
- `apps/cli/main.cpp:1143-1205` — sampled speculative step: `st0`, verify forward, row loop, `samp_seconds += st1 - st0` (the overlap).
- `apps/cli/main.cpp:1225-1230` — off-path `samp_seconds` (the honest 11-13 ms timer).
- `src/models/sampling.cpp:233-345` — `apply_sampling_cuts`: threshold scan, radix/partial sort, keep mask; ~0.26 ms/row at 128k vocab.
- `src/models/sampling.cpp:64-73` — `softmax_probabilities` AVX2 path; ~0.08 ms/row.
- `reports/idea-invented-sampled-spec-8k-measured.md` — the report corrected here.
- `reports/idea-invented-sampled-spec-2k-measured.md` — its speed result stands; its sampler attribution is flagged unverified.

## 6. Reimplementation recipe

1. Add `double samp_softmax_seconds = 0, samp_cut_seconds = 0; long long samp_rows = 0;` beside `samp_seconds`.
2. In the `penalized_softmax` lambda, time `softmax_probabilities` (increment softmax + rows) and, inside the cut branch, time `apply_sampling_cuts` (increment cut).
3. Print `sampling : softmax X ms over N rows, cut Y ms` after the `timers` line.
4. Build + `tge_tests` (171/171). Run:
   `tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt "<fox x2048>" --context 8704 --n 32 --kv f16 --threads 10 --gpu off --temp 0.8 --top-p {0.9|1.0} --seed 7 --speculative`
   Expect softmax ~2.1-2.5 ms over 29-31 rows, cut 7.55 ms at top-p 0.9 and 0.00 at 1.0; the old `timers sample` still reads ~1.6-2.0 s and must be ignored for the sampled path.
5. Revert `apps/cli/main.cpp`, rebuild, retest, confirm clean tree. No code ships.

## 7. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56)
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed (spike build)

8k sampled spec, ctx 8704, n 32, f16, 10 threads, temp 0.8, seed 7, --speculative:
  top-p 0.9 -> 8.17 t/s; 7 steps, 3.29 acc/step; timers sample 2022.9 ms (53.3%);
               sampling softmax 2.46 ms / 29 rows, cut 7.55 ms
  top-p 1.0 -> 11.06 t/s; 6 steps, 4.33 acc/step; timers sample 1610.1 ms (57.5%);
               sampling softmax 2.14 ms / 31 rows, cut 0.00 ms
git checkout -- apps/cli/main.cpp                 -> clean
cmake --build --preset release -> clean; tge_tests -> 171/171; git status -> clean
python aikit.py check aegean-tge                  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
