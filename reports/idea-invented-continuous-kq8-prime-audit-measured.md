# MEASURED: prefix-prime admission path under path-invariant kq8 continuous decode — transcripts identical, primes engaged, +5-12% throughput

- **Question**: The closeout's regression set covers occupancy (solo vs busy) and generate parity, but not the **cross-sequence prefix prime** path in `bench continuous`: when a request's whole prompt is already in the multi-prefix table, the admission does `prime_cache_from_table` and feeds only the last prompt token (`prompt_skip = prompt_tokens - 1`). Under the path-invariant change, that 1-token remainder now takes the kq8 batch kernel instead of the float batch kernel. Does the prime path still produce identical transcripts across arms and occupancies, and does the kq8 win survive when admission is primed?
- **Cycle date**: 2026-11-11
- **Verdict**: **MEASURED — clean.** `qwen2.5-0.5b-instruct-q4_k_m`, `bench continuous --threads 10 --kv f16`, prompt-tokens 24 and 56, `--tokens 8` audit / `--tokens 16` throughput:
  - **Primes engaged**: the `[cow-mp]` line count is 1 for the solo run (populate only) and **12 for the busy run** (1 populate + 11 primes) at both prompt lengths — the admission-skip path is exercised, not bypassed.
  - **Audit passes**: pt=24 -> `CA-C6-CC…` identical in all four arm/occupancy combinations (float solo/busy, kq8 solo/busy); pt=56 -> `DD-EA-7F…` identical in all four.
  - **Throughput under the primed workload** (12 requests, 8 slots, pt=24, 16 tokens): float 148.02/153.87 t/s -> path-invariant 155.28/172.30 t/s, ratios **1.049/1.120** (median ~1.08x); `state: independent (all transcripts identical)` in all runs.
- **Tree state at close**: spike reverted from `src/models/qwen2.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

- Admission skip (`apps/cli/main.cpp:1630-1671`): if `shared_prefix_len(prompt) >= prompt_tokens` and `prime_cache_from_table(slot, prompt, prompt_tokens)` succeeds, the batch feeds from `index = prompt_tokens - 1`, producing logits at row `active_count + (prompt_tokens - prompt_skip) - 1`.
- With the path-invariant edit, that single fed row goes through `gemv_dequant_batch_kq8_avx2` at `tokens >= 1` instead of the float batch kernel. The primed rows themselves are memcpy'd from the table and are kernel-independent.
- The audit checks that a request decoded alone (which populates the table and then hits `prompt_tokens` rows) and the same request decoded in a busy batch (which primes) yield the same output under each arm, and that the arms agree.

## 2. Results

### Audit (first-transcript hash, `--tokens 8`)

| prompt-tokens | arm | solo | busy8 | match |
|---:|---|---|---|---|
| 24 | float | `CA-C6-CC` | `CA-C6-CC` | yes |
| 24 | kq8 path-invariant | `CA-C6-CC` | `CA-C6-CC` | yes |
| 56 | float | `DD-EA-7F` | `DD-EA-7F` | yes |
| 56 | kq8 path-invariant | `DD-EA-7F` | `DD-EA-7F` | yes |

`[cow-mp]` lines: solo 1 (POPULATE), busy8 12 (POPULATE + 11 PRIME) at both lengths. The ids themselves are `34319 77 1699 2 2585 311 1855 264…` at pt=24 in every combination.

*Hash caveat*: `first:` prints only the first request's `--tokens` ids, so hashes from different `--tokens` values are not comparable (the earlier pt=24 audit used 16 tokens and therefore a different hash for the same ids). Compare hashes only within one `--tokens` setting.

### Throughput under the primed workload

| pair | float t/s | path-invariant t/s | ratio |
|---:|---:|---:|---:|
| 1 | 148.02 | 155.28 | 1.049 |
| 2 | 153.87 | 172.30 | 1.120 |

12 requests, 8 slots, prompt-tokens 24, 16 decode tokens.

## 3. Reading

- The prime path does not weaken the path-invariance argument: the row that is actually computed (the last prompt token) now uses kq8 in both solo and busy runs, and the memorised rows are copies. Transcripts agree across arms and occupancies.
- The kq8 win survives priming (+5-12% on the 0.5B at this shape; the 3B numbers remain the headline).
- This closes the last known continuous surface in the closeout's regression set: occupancy without priming (0.5B 8/24/48, 3B 24/48/56), occupancy with priming (this cycle, 24/56), generate parity (both models), and in-run transcript checks.
- Promotion guidance is unchanged: `--kq8-batch` opt-in first; the pre-merge checklist should add the primed audit above (compare `[cow-mp]` counts so a non-primed run cannot silently pass).

## 4. Reimplementation recipe

1. Apply the path-invariant change (`reports/idea-invented-continuous-kq8-path-invariant-keep.md` §8) and build.
2. Audit: `bench continuous --model <0.5B q4_k_m> --requests {1,12} --tokens 8 --threads 10 --sequences 8 --kv f16 --prompt-tokens {24,56}`; require a `[cow-mp]` count of 12 on the busy run and hash equality across all four arm/occupancy combinations at each prompt length.
3. Throughput: `--requests 12 --tokens 16 --sequences 8 --prompt-tokens 24`, alternating arms; expect +5-12% on 0.5B.
4. Revert (report-only) or include in the promotion evidence.

## 5. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge bench continuous ... --requests {1,12} --tokens 8 --sequences 8 --kv f16 \
  --prompt-tokens {24,56}                                         -> audit table + cow counts
tge bench continuous ... --requests 12 --tokens 16 --sequences 8 --kv f16 \
  --prompt-tokens 24                                              -> 1.049/1.120
git checkout -- src/models/qwen2.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. Baseline binary lives at `build/idea/tge-cont-float-base.exe`; the committed tree is unchanged except this report.
