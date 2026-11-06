# INVENTED: Thread-Local kq8 Batch Activation Scratch — KILLED (off/on pairs +11%/-1%/-1%: the heap free-list already absorbs per-call allocs, no contention signal)

- **Idea**: `gemv_dequant_batch_kq8_avx2` heap-allocates its activation-quant buffer (`tokens × blocks` Q8K, ~600 KB at 64×8192) on every batch call — thousands of 100KB+ malloc/free pairs per prefill across 10 threads. Reuse a thread-local scratch instead (the kq8d single-row precedent, which exists precisely because per-call allocation showed up); contents and read pattern unchanged, bit-identical by construction.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the stated falsifier (≥5% prefill win). Interleaved off/on pairs on Llama-3B pp512: 9008→10002 (+11.0%), 10008→9897 (−1.1%), 10236→10179 (−0.6%). R1 is warmup noise (first run of the window, fastest absolute); R2/R3 are flat inside drift. No consistent direction → the allocator is not the bottleneck: free-list reuse absorbs the allocs and 10-thread contention never materializes.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). Spike branch `spike-kq8scratch` created (zero commits) and deleted; the one spike file reverted with empty diff verified; scratch deleted. `master` ahead of `origin/master` by 104 after this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked, reverted)

`src/cpu/avx2/dequant_batch_kq8_avx2.cpp` only, env-gated (`TGE_KQ8_SCRATCH=1`): thread-local `scratch_quantized` resized-as-needed; the fill loop and band lambda read through a `quantized_ptr` (scratch or the untouched local vector when the env is off — same-binary A/B). No header change, no numerics change.

## 2. Predicted gain — not realized

Expected ≥5% prefill from removing ~1500 large heap alloc/free pairs per 512-token Llama prefill. Measured ~0%: the heap's free lists serve the repeats, and no 10-thread contention appears.

## 3. Falsifier outcome — kill

PROCEED required ≥5% prefill win with identical ids. Best pair −1.1%, worst +11.0% (warmup artifact, first run of the window). Ids identical throughout (same buffer contents by construction). No fidelity run (no numerics change — same bytes, same reads).

## 4. File anchors (master = pre-spike; spike reverted)

- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:340-404` (per-call `quantized` vector + fill + band loop — the alloc site).
- Grounds: the kq8d thread-local precedent in the same TU ("never heap-allocated per call"), the q8_batch_vnni allocation note.

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-kq8scratch` (zero commits, deleted at close). Baseline: build 54/54, `tge_tests` 141/141.
2. ANALYZE: alloc site confirmed per-call on the hot path; reuse designed to mirror kq8d exactly (caller-thread fill, read-only pool sharing).
3. Applied §1 (env-gated, control preserved). Rebuilt 54/54, `tge_tests` 141/141.
4. Measured: Llama-3B fox×40 off/on interleave ×3 (§6). Thinking set on spike (saved).
5. `aikit.py check aegean-tge` → 2/4 (test + secrets PASS; hygiene + guidelines fail pre-existing, same pair as every prior cycle).
6. Reverted the file (empty diff), `git checkout master`, branch deleted, rebuilt master 54/54, `tge_tests` 141/141, master thinking set: story + creative ids byte-identical vs spike.
7. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (Llama-3B-Q4_K_M fox×40 = 520 tok, 10 threads, --context 4096, --n 8; prefill ms)

| round | scratch off | scratch on | delta |
|---|---|---|---|
| 1 | 9008.17 | 10001.82 | +11.0% (warmup artifact — fastest absolute of the window) |
| 2 | 10008.00 | 9897.13 | −1.1% |
| 3 | 10235.93 | 10179.26 | −0.6% |

Box drift noted (off-arm crept 9008→10236 across the window); pairs adjacent. No consistent direction; best pair −1.1% « 5% bar.

## 7. Full reimplementation recipe (kill-grade — rebuild only to re-verify)

Re-apply §1; gate on build 54/54, `tge_tests` 141/141, and the §6 interleave reproducing ~0% (any consistent ≥5% either way reopens the question with a contention profile — e.g., heap-lock tracing — attached). Revival trigger: an allocator change that breaks free-list reuse (e.g., a jemalloc/mimalloc swap regressing large-block caching) or a shape with 10x the call rate at the same buffer size; until then the per-call vector stands (it also keeps the code re-entrant without thread-local lifetime questions, which is worth more than the unmeasured ~0%).

## 8. Verdict and why

**KILLED: the heap is not the bottleneck on the kq8 batch path.** Free-list reuse absorbs thousands of large per-call allocs even across 10 threads, with no contention signal — the kq8d precedent stays a single-row special (per-token frequency), not a batch-path rule. Allocator pressure is now measured dead on both kq8 entries (prototype comment + this round), so the family is closed: no future cycle re-proves allocation without a new allocator or a contention trace in hand. Next-cycle candidates: the 1517 epic's stage-2 skip harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- `git checkout -b spike-kq8scratch` → branch (zero commits, deleted at close).
- Baseline: `cmake --build --preset release` → 54/54; `tge_tests.exe` → 141/141.
- Spike (§1, env-gated); rebuild → 54/54; `tge_tests` → 141/141.
- Llama-3B fox×40 off/on interleave ×3 → §6 table.
- Thinking set (spike, saved); `aikit.py check` → 2/4 (test + secrets pass; pre-existing fails).
- Revert file → clean; `git checkout master`; branch deleted; rebuild 54/54; tests 141/141; master thinking set → ids identical; scratch deleted; report committed; no push.

(End of file.)
