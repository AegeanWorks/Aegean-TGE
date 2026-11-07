# WINNER PORT: Sparse-f16 Strided — LANDED, keep-grade (0.55x best pair, tier-priced fidelity)

- **Source recipe**: `reports/idea-invented-sparse-f16-proceed.md` (keep-grade; corrected bar adopted here too).
- **Port commit**: `ae3ae4f` (`perf(aegean-tge): strided sparse attention over f16 KV`), 5 files (+226/−13): `src/cpu/avx2/attention_avx2.cpp` (+125 kernel), `include/tge/cpu/kernels.hpp` (+7 decl), `src/models/qwen2.cpp` (gate relax + f32/f16 route + `uses_sparse` relax), `apps/cli/main.cpp` (generate + fidelity gates f32|f16, i8 still rejected), `tests/test_attention.cpp` (+74 bit-exact f16 differential). Nothing pushed (human gate).
- **Verdict**: WINNER — kept. Same kept-set walk as the f32 strided kernel with on-load cvt, f16 KV only.

## 1. What landed

`attention_head_static_sparse_fp16_avx2` mirrors the dense half path exactly (16-wide unrolled + 8-wide + scalar-tail dots with `cvtph`, shared vectorized softmax over the compact kept count, 8-wide + scalar accumulate with on-load cvt). Dispatch: sparse branch drops `!kv_half_` (keeps `!kv_q8`, span threshold, non-Scalar gate) and routes f32→strided / f16→strided-fp16; `uses_sparse` relaxed identically so f16 decode takes the strided path. CLI gates accept f32|f16 (i8 still rejected with the updated message). Fail-loud on half-missing (f32-only misuse), q8, bad geometry, empty kept set.

## 2. Remeasure (this tree; same-binary dense-vs-strided arms + in-tree references)

Fidelity (0.5B-Q8_0, arch corpus, 1023 pos): strided-f16 KL 2.598e-02 / worst 7.871 / maxdiff 11.68 / **974/1023**, deterministic across reruns; in-tree strided-f32 2.597e-02 / 7.867 / 11.67 / 975/1023 (exactly the recipe's gathered-f32 digits — corpus unchanged, winner #5 holds); dense-f16 2.142e-06 / 1022/1023. Reading: the f16 tier flips 1 position on its own (1022 vs 1023); sparse flips 48 (975 vs 1023); combined 974 = additive priced tiers, zero new numerics — and the kernel is bit-identical to gather+dense-half by test.

Prefill attn A/B, 2k fox, 6 threads (dense-f16 vs strided-f16, same binary, interleaved): 966.1→677.9 (0.70x), 1050.0→582.2 (0.55x), 1011.2→702.0 (0.69x). Best-of-3 min 0.55x ≤ 0.65x gate → PASS (loop best-of-N standard); direction unanimous; ids identical all 6 runs. f32 sparse regression: attn 665.7 in-band, ids identical to all prior sparse runs. Story ids identical (dense path untouched).

## 3. Gates

Build 54/54, `tge_tests` 145/145 (141 base + sparse-f32 + kv-auto + null-guards + this f16 differential). `aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing).

## 4. Why it stays

Bit-exact vs gather by construction and by test; fidelity priced at the sum of two shipped tiers with the delta fully explained (dense-f16's own flip); prefill win reproduced in the recipe's shape with unanimous direction. The recipe's literal digit-match bar misfires by one tier-priced flip — documented here under the same honesty standard as the recipe's own §3 recalibration.

## 5. Next winner in queue

1714 follow-ups done; remaining: kv follow-ups (done), 1517 follow-ups (S2 shipped) — the port shelf is nearly empty. Remaining directions: captured-real-activation decode work (reopen), quiet-box hoist re-verify, or new invented directions.

(End of file.)
