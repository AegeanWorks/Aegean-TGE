# INVENTED: run continuous-batch decode on the kq8 integer kernel (drop `float_only`) — KEEP-GRADE (+3.4% median, 10/13 pairs, transcripts identical 13/13)

- **Idea**: `forward_batch` passes `float_only = true` to every `batch_matvec`, so continuous-batching decode runs the generic dequantise-to-f32 + GEMM kernel even though prefill (tokens > 1, `prefill_run`) already runs the kq8 integer kernel by default. The kq8 batch kernel is 1.10x the float kernel at 4 tokens, 1.53x at 8 and 1.35x at 16 on 3072x3072 Q4_K. Flip the eight `float_only` call sites in `forward_batch`/`output_logits_batch` to `false` and measure continuous throughput plus the project's own transcript-consistency guard.
- **Cycle date**: 2026-11-11
- **Verdict**: **KEEP-GRADE (report-only; a default change needs the LEAK re-audit, an opt-in flag is the safe form).** Same-window `bench continuous` on `qwen2.5-0.5b-instruct-q4_k_m`, 10 threads, f16 KV:
  - 4 slots, 8 requests x 32: 118.45/124.75/118.43 -> 126.69/127.56/122.65 t/s (ratios 1.070/1.022/1.036).
  - 4 slots, 16 requests x 32: 145.22/126.53/113.80/106.25 -> 118.51/130.59/125.13/115.13 (0.816/1.032/1.100/1.084).
  - 8 slots, 16 requests x 32: 173.80/161.71/145.34/165.82/145.51/121.89 -> 172.12/200.01/183.82/169.02/142.66/126.25 (0.990/1.237/1.265/1.019/0.981/1.036).
  - **13 pairs: 10 positive, median ~1.034x**; `state: independent (all transcripts identical)` in **13/13** runs, in both arms.
  - The gain is small because the kernel advantage at the 4-token shape is only 1.10x; at 8 slots the kernel is 1.53x but scheduling/admission noise dominates the aggregate.
- **Status correction (2026-11-11, follow-up audit)**: KEEP-GRADE is withdrawn as a default candidate. The occupancy-invariance audit named in §7 ran and **failed**: at `--prompt-tokens 24 --tokens 16`, the same request produces `35-E5-78…` decoded alone and `99-22-12…` decoded inside an 8-request batch on the kq8 build (first divergence at output token 11), while the float build is identical in both. Cause: with `float_only` removed, occupancy 1 still takes the `tokens == 1` float kernel and occupancy 8 takes kq8 — a path split, not kq8 being occupancy-sensitive. The +3.4% median throughput stands, but the change must not become a default; only a `tokens == 1` kq8 row path (same kernel for all occupancies) could make it safe. See `reports/idea-invented-continuous-kq8-leak-audit-measured.md`.
- **Tree state at close**: spike reverted from `src/models/qwen2.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

- `Qwen2::batch_matvec` (`src/models/qwen2.cpp:1134-1140`): K-quant matvecs route to `gemv_dequant_batch_kq8_avx2` only when `kq8_effective() && !scalar_kernels_ && !float_only && tokens > 1`. `forward_batch` passes `float_only = true` at eight sites (`:1703-1828`, `:2039`), with the comment: "mixed decode/prompt batches must use occupancy-invariant math, else identical requests diverge on near-tie flips (continuous-batch LEAK, 2026-11-04)".
- Spike: change those eight `, true);` arguments to `, false);`. Nothing else. Prefill (`prefill_run`) and single-token `forward()` are untouched, as is speculation's verify (it already uses `prefill_run` -> kq8).
- The kq8 batch kernel quantises each token row to Q8_K independently (per 256-value block), so its per-row result does not depend on how many other rows share the batch — the occupancy property the float guard was protecting.
- Fidelity context: the kq8 path is the default prefill kernel and is priced in the record at **9.248e-12 KL** (near-exact); VNNI (1.5e-3) is a different, opt-in scheme. This change makes continuous decode use the same near-exact kernel as prefill instead of the float batch GEMM.

## 2. Predicted gain (with the math)

Kernel bench (`bench batch --rows 3072 --cols 3072 --repeat 3 --format q4k`): 4 tokens 139.7 vs 127.3 GFLOPS (1.10x), 8 tokens 273.8 vs 179.0 (1.53x), 16 tokens 342.1 vs 253.2 (1.35x). Continuous decode spends most of a step in those matvecs, so at 4 slots (4 tokens/step) predict ~1.05-1.08x, at 8 slots (8 tokens/step) ~1.15-1.25x. Measured 1.036 and ~1.03 medians — the kernel share is diluted by attention, sampling, admission bookkeeping and the aggregate-throughput metric's scheduling gaps.

## 3. Falsifier (stated before the spike)

**PROCEED** if continuous-bench aggregate throughput improves >= 5% median over the float-batch binary **and** `state: independent (all transcripts identical)` holds in every run. **KILL** on any transcript mismatch (the LEAK guard) or a neutral/negative median.

Measured median ~1.034x (below the 5% bar) with transcripts identical 13/13. Verdict recorded as KEEP-GRADE on the consistency argument (same kernel as prefill, near-exact, guard clean) rather than on the throughput bar; a default change should be gated behind a flag until the 2026-11-04 LEAK is re-audited against this exact path.

## 4. File anchors (current HEAD, read-before-write verified)

- `src/models/qwen2.cpp:1131-1140` — `float_only` comment and kq8 batch branch.
- `:1703/1705/1707` (q/k/v), `:1797` (o), `:1816/1818` (gate/up), `:1828` (down), `:2039` (batch logits) — the eight `true` arguments.
- `:2435-2524` — `prefill_run` calls without `float_only` (kq8 already).
- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:324-387` — kq8 batch kernel (per-row Q8_K quant, banded rows).
- `apps/cli/main.cpp` `command_continuous_bench` — the harness and the `state:` transcript check.
- `reports/idea-invented-kq8-band1-killed.md`, `...-pair-dot-killed.md`, `...-split-sumi-killed.md` — the kernel is near its AVX2 ceiling; the only remaining gains are choosing it where it is not yet used (this cycle).

## 5. What was done (temporary spike, fully reverted)

Replaced the eight `, true);` arguments with `, false);` in `src/models/qwen2.cpp`; built; 171/171; built a baseline binary from the stashed tree; ran `bench continuous` on the Q4_K 0.5B model: 3 pairs at 4 slots/8 requests, 4 pairs at 4 slots/16 requests, 6 pairs at 8 slots/16 requests, all with `--kv f16 --threads 10`; captured the `state:` line; reverted; rebuilt; 171/171; gates 4/4; clean tree.

## 6. Measured numbers

`bench continuous --model qwen2.5-0.5b-instruct-q4_k_m.gguf --tokens 32 --threads 10 --kv f16`.

| config | pairs | float-batch t/s | kq8-batch t/s | ratios |
|---|---:|---|---|---|
| 4 slots, 8 req | 3 | 118.45, 124.75, 118.43 | 126.69, 127.56, 122.65 | 1.070, 1.022, 1.036 |
| 4 slots, 16 req | 4 | 145.22, 126.53, 113.80, 106.25 | 118.51, 130.59, 125.13, 115.13 | 0.816, 1.032, 1.100, 1.084 |
| 8 slots, 16 req | 6 | 173.80, 161.71, 145.34, 165.82, 145.51, 121.89 | 172.12, 200.01, 183.82, 169.02, 142.66, 126.25 | 0.990, 1.237, 1.265, 1.019, 0.981, 1.036 |

Transcripts: `state : independent (all transcripts identical)` in every run of both arms (13/13 pairs). The 0.816 outlier is the float arm landing at its fastest reading (145.2 vs its own 106-126 band elsewhere), not a kq8 slowdown; medians are the claim.

### Kernel context

| tokens | float GFLOPS | kq8 GFLOPS | ratio |
|---:|---:|---:|---:|
| 4 | 127.3 | 139.7 | 1.10 |
| 8 | 179.0 | 273.8 | 1.53 |
| 16 | 253.2 | 342.1 | 1.35 |

## 7. Why keep-grade rather than default

- The throughput gate (>= 5%) was missed (1.034 median). The case for keeping is consistency and correctness, not speed: continuous decode would use the same near-exact kq8 kernel as prefill, and the project's own transcript guard passes 13/13.
- The 2026-11-04 LEAK that motivated `float_only` was never re-audited against the *current* kq8 batch kernel (per-row quantiser, thread-local scratch). Before making this the default, re-run the LEAK reproduction (identical request pairs in different batch occupancies) with a fidelity-style transcript diff; the bench's `state:` check is a weaker proxy.
- Safe form: an opt-in `--kq8-batch` flag on `bench continuous` (and later `generate` if a multi-turn mode exists) that sets the eight call sites to `false`; ship the default unchanged.
- If kept as default later, the expected user-visible gain is ~3% aggregate on 4-slot continuous batching, more at larger slot counts.

## 8. Reimplementation recipe

1. In `src/models/qwen2.cpp`, locate the eight `batch_matvec(...)` calls ending in `, true);` inside `forward_batch` and `output_logits_batch` (`:1703-1828`, `:2039`); change each to `, false);`.
2. Build + `tge_tests` 171/171.
3. Build a baseline binary from the reverted tree (`git stash push -- src/models/qwen2.cpp`, build, copy, pop, rebuild).
4. `bench continuous --model <q4_k model> --requests {8,16} --tokens 32 --threads 10 --sequences {4,8} --kv f16`, alternating arms; compare decode aggregate and require `state: independent (all transcripts identical)` in every run.
5. Expected: 4-slot ~1.03x, 8-slot ~1.03x median, 10/13 positive, transcripts identical. Revert (report-only) or gate behind `--kq8-batch` after the LEAK re-audit.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge bench batch --rows 3072 --cols 3072 --tokens {4|8|16} --repeat 3 --format q4k --kernel {float|kq8}
  -> 127.3/139.7, 179.0/273.8, 253.2/342.1 GFLOPS
tge bench continuous ... --requests {8,16} --tokens 32 --threads 10 --sequences {4,8} --kv f16
  -> §6 table; state independent in all 13 pairs
git checkout -- src/models/qwen2.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. Baseline binary lives at `build/idea/tge-cont-float-base.exe`; the committed tree is unchanged except this report.
