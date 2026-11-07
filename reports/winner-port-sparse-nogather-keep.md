# WINNER PORT: No-Gather Strided Sparse — LANDED, keep-grade (0.44x attn, bit-exact)

- **Source recipe**: `reports/idea-invented-sparse-nogather-proceed.md` (+ nogather-prefill-proceed, sparse-f16-proceed as staged follow-ups, NOT this port).
- **Port commit**: `1c11771` (`perf(aegean-tge): no-gather strided sparse attention replaces gather+dense`), 4 files (+192/−26): `src/cpu/avx2/attention_avx2.cpp` (+104 kernel), `include/tge/cpu/kernels.hpp` (+11 decl), `src/models/qwen2.cpp` (+4/−26 dispatch swap), `tests/test_attention.cpp` (+69 bit-exact differential). Nothing pushed (human gate).
- **Verdict**: WINNER AGAIN on remeasure — kept as the sparse path. Arithmetic identical by construction; only traffic removed.

## 1. What landed

`attention_head_static_sparse_avx2(args, sink, window, stride)`: the dense per-head computation (dot/scale/max, shared vectorized softmax, weighted accumulate — op order untouched) over the sink/stride/window kept set read in place with the native kv_stride. Kept positions ascend exactly as `build_sparse_index` emits them (verified segment-by-segment), so outputs are bit-identical to gather+dense; `args.span` keeps the full-span contract, scores[0..kept) hold the compact weights. Throws fail-loud on half/q8/bad geometry (f32-only, like the old path). Dispatch: same gate bit-for-bit, gather block deleted.

## 2. Remeasure (this tree; baseline = pre-port gathered binary, interleaved best-of-3)

0.5B-Q8_0 fox×200 (2000 tok), 6 threads, `--sparse` defaults (sink 16/window 512/stride 4):

| pair | gathered attn | strided attn | ratio |
|---|---|---|---|
| 1 | 1876.6 ms | 719.0 ms | 0.38x |
| 2 | 1772.0 ms | 809.7 ms | 0.46x |
| 3 | 1545.2 ms | 817.7 ms | 0.53x |

Prefill totals move less (matvec+norm dominate at 2k: 55-65%), but the changed path reproduces the recipe ratio in every pair; decode attn drops similarly (42→30/35/33). Per-run ids byte-identical all 3 pairs (`576 3974 13876 ...`).

## 3. Gates

Build 54/54, `tge_tests` 142/142 including the new `static_sparse_matches_gather_bit_exact` differential (spans 600/1024/2048, exact `==`). Fidelity `--sparse` on port: KL 2.597e-02 / worst 7.867 / maxdiff 11.67 / 975-1023 — digit-matches the gathered tier. Thinking set (dense path): story + haiku ids byte-identical to every prior run. `aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing).

## 4. Why it stays

Third independent reproduction of the recipe ratio (spike + re-verify + this port), with bit-exactness proven by construction AND by test AND by ids. Zero dense-path impact (opt-in flag only, same gate).

## 5. Next winner in queue

kv-auto / 1714 hardening / 1517 stage-2 follow-ups — same treatment: baseline binary saved first, implement, remeasure, keep on repeat win.

(End of file.)
