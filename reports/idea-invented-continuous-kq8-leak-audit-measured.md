# MEASURED: occupancy-invariance (LEAK) re-audit for continuous-batch kq8 decode — FAILED at prompt-tokens 24 (solo vs busy diverge at output token 11); keep `float_only`

- **Question**: `idea-invented-continuous-kq8-batch-keep.md` kept the kq8-batch decode change as report-only and named its precondition: re-run the 2026-11-04 continuous-batch LEAK reproduction on the current per-row-quantised kq8 kernel before any default. This cycle runs that audit: the same request decoded alone vs the same request decoded inside an 8-request batch must produce identical output. If it does, the guard's premise no longer holds; if it does not, the guard is vindicated.
- **Cycle date**: 2026-11-11
- **Verdict**: **LEAK REPRODUCED — the guard must stay.** With the eight `float_only` call sites flipped to `false` (the previous cycle's exact spike), on `qwen2.5-0.5b-instruct-q4_k_m` at `--prompt-tokens 24 --tokens 16 --threads 10 --kv f16`:
  - float-batch binary: solo `99-22-12…`, busy8 `99-22-12…` — **identical** (as shipped).
  - kq8-batch binary: solo `35-E5-78…`, busy8 `99-22-12…` — **different**, reproducibly (two trials, deterministic hashes).
  - First divergence at output token **11 of 15**: solo `…501 3250 304 431 271 641 431 11` vs busy `…501 3250 304 264 38228 304 431 271`.
  - Mechanism: with `float_only` removed, a request at occupancy 1 still takes `tokens == 1` -> the float batch kernel, while the same request at occupancy 8 takes `tokens > 1` -> the kq8 integer kernel. The two kernels are near-exact but not bit-identical, so the same request changes output depending on how many neighbours share the step — the exact composition-dependence the guard exists to prevent. It is a **path split**, not kq8 being independently occupancy-sensitive.
  - Two other shapes happened not to diverge: prompt-tokens 8 and 48 produced identical solo/busy hashes on both arms (degenerate/repetition-dominated outputs). One divergent shape is enough to fail the audit.
- **Correction to the prior report**: `idea-invented-continuous-kq8-batch-keep.md` recorded KEEP-GRADE with this audit pending. The audit now fails, so that verdict downgrades to **KILLED as a default candidate; keep-grade only as an explicit opt-in for single-occupancy batches** (where `tokens > 1` never arises), and even that needs the divergence acknowledged. A corrected-status line was appended to that report.
- **Tree state at close**: spike reverted from `src/models/qwen2.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Setup

- Binary A (float batch, shipped): `build/idea/tge-cont-float-base.exe`, built from HEAD with the eight call sites at `true`.
- Binary B (kq8 batch): HEAD with the eight `, true);` arguments at `:1703-1828`, `:2039` changed to `, false);`.
- Harness: `bench continuous --model qwen2.5-0.5b-instruct-q4_k_m.gguf --requests {1,16} --tokens {16,32} --threads 10 --sequences {4,8} --kv f16 --prompt-tokens {8,24,48}`.
- Test: the first request's transcript (`first :` line) in a solo run (1 request) must equal the first request's transcript in a busy run (16 requests, 8 slots), byte for byte. Requests are identical, so any difference is composition-dependence.

## 2. Results

| prompt-tokens | arm | solo first hash | busy8 first hash | match |
|---:|---|---|---|---|
| 8 | float | `05-74-D3` | `05-74-D3` | yes |
| 8 | kq8 | `05-74-D3` | `05-74-D3` | yes |
| **24** | float | `99-22-12` | `99-22-12` | yes |
| **24** | **kq8** | **`35-E5-78`** | **`99-22-12`** | **NO** |
| 48 | float | `10-3D-0F` | `10-3D-0F` | yes |
| 48 | kq8 | `10-3D-0F` | `10-3D-0F` | yes |

pt=24 repeated twice per arm: kq8 solo `35-E5-78` twice, kq8 busy `99-22-12` twice — deterministic, not noise. (`state : independent (all transcripts identical)` still reports true inside each run because the in-run sequences all see the same occupancy path; the leak is across runs, which is why the audit uses solo-vs-busy, not the bench's in-run check.)

Divergence slices (0-based, tokens 8-15):

- solo (kq8): `501 3250 304 431 271 641 431 11`
- busy8 (kq8): `501 3250 304 264 38228 304 431 271`

First differing index: **11**.

## 3. Reading

- The guard is correct and its comment is accurate: mixed decode/prompt batches change the kernel *path* (tokens==1 float vs tokens>1 kq8), and that path split is observable in the output at a near tie. Flipping `float_only` would make a request's tokens depend on load — unacceptable for a server.
- The prior keep report's throughput finding (+3.4% median, transcripts identical 13/13) stands as measured, but its in-run `state` check was too weak to detect cross-run composition effects. The audit method here (solo vs busy, same request) is the right gate and should be the template for any future batch-path change.
- The only safe form of the change is one that removes the path split itself: make `tokens == 1` use the *same* kq8 kernel (the single-row kq8 path already exists — `gemv_dequant_kq8_row_avx2`), so occupancy 1 and occupancy N agree bit-for-bit. That is a new spike with its own per-row exactness check (kq8 row vs kq8 batch at M=1), not a guard flip.
- Until then: `float_only` stays `true` in `forward_batch`; continuous decode keeps the float batch kernel; the kq8 batch kernel remains prefill-only.

## 4. Reimplementation recipe (re-run the audit)

1. Build binary A from HEAD (as shipped) and binary B with the eight `forward_batch`/`output_logits_batch` `float_only` arguments set to `false`.
2. For prompt-tokens in {8, 24, 48}: run each binary with `--requests 1 --sequences 4` and `--requests 16 --sequences 8`, same `--tokens 16 --threads 10 --kv f16`.
3. Hash the `first :` id line of each run (SHA-256 prefix is enough). Match solo vs busy per arm.
4. Expected: float arm matches on all shapes; kq8 arm diverges at one or more shapes (pt=24 here, first diff token 11). Any divergence fails the audit; do not flip the guard.
5. To pursue the change anyway, first build the `tokens == 1` kq8 row path into the batch entry (so the kernel is path-invariant), then re-run this audit until solo and busy match on every shape.

## 5. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge bench continuous ... --requests {1,16} --tokens 16 --threads 10 --sequences {4,8} \
  --kv f16 --prompt-tokens {8,24,48}                              -> §2 table
git checkout -- src/models/qwen2.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. Binary A lives at `build/idea/tge-cont-float-base.exe`; the committed tree is unchanged except this report and the correction line in the prior one.
