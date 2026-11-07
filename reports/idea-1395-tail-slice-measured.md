# IDEA-1395 (slice: exact-fp32 KV tail over i8) — MEASURED: quality wins 10x, decode at parity; keep-grade recipe filed, spike reverted

- **Idea ID**: IDEA-1395 (dossier `kilo-sweep-2026-11-03.md` §IDEA-1395: variance-normalized KV + precision tail; sliced to the exact-tail half — last-N positions exact fp32 over the i8 bulk, single softmax — because the KVarN-Hadamard half needs a sub-8-bit bulk that does not ship, and the i8-bulk claim in the dossier is already shipped as `--kv i8`).
- **Cycle date**: 2026-11-07
- **Verdict**: MEASURED — quality arm wins decisively (mean KL 2.133e-4 → 2.087e-5, 10.2×; top-1 196/199 → 199/199); decode arm at parity (best-of-3 warmed reverse-order 8.50 vs 8.03 t/s, +5.8%, inside ±20% box noise; attn-phase ms indistinguishable). Free-running ids diverge from token 8 (expected for lossy KV; KIVI/VNNI precedent). Keep-grade: new quality/bytes point between i8 and f16 at ~+1% decode cost. Spike fully reverted; only this report committed.
- **Tree state at close**: `master`, `git status` clean, spike branch created and deleted with nothing committed on it, stash untouched (pre-existing peer stash 2026-11-04 only). Final gates on master: build clean, `tge_tests` 141/141. Nothing pushed (human gate).

## 1. Mechanism (spiked slice)

Alongside the i8 store, keep exact fp32 K/V for the last N positions (ring slot `pos % N` per sequence/layer, `TGE_KV_TAIL=N`, default off). Score and value loops read exact rows for absolute positions `>= span - N`, i8-dequant otherwise, merged in the single existing softmax — the beellama `--kv-tail-tokens` arrangement, minus KVarN (no Hadamard anywhere on tree; verified by grep) and minus FlashAttention (tree's own vectorised kernels). Head kernel (`attention_head`, span = position+1, rows contiguous from 0) and GQA chunk kernel (chunk-relative tokens, absolute position = `tail_start + token`, `tail_first = span - min(span,N)` passed by the caller) both branched, scalar + AVX2 each (4 kernels × score/value = 8 branch sites, one shared pattern).

Predicted gain: KL between flat i8 (2.133e-4 @ctx512 here) and f16 (1.658e-6), because recent-position error dominates next-token logits; decode cost ≈ +1% (attn is ~4-11% of decode; tail adds fp reads for N=64 rows while skipping their q8 rows: +13% KV-attn traffic ≈ +0.5-1.4% decode, plus ~0.7 ms/token store memcpy).

## 2. Falsifier (from the dossier, adapted to the slice)

Dossier: median KLD > 2e-3 or flips > 7/497 → kill (retraining reject N/A — no retraining here). Slice gates: (a) fidelity mean KL not better than flat i8 at matched protocol, or top-1 worse → kill, no bench; (b) warmed best-of-3 decode regresses > 5% vs flat i8 → kill on cost. Neither tripped.

## 3. File anchors (current `master`, untouched after revert)

- `include/tge/cpu/kernels.hpp` — `AttentionArgs`/`AttentionGroupArgs` (spike added `tail_n/tail_first/tail_start/tail_keys/tail_values`; reverted).
- `include/tge/models/qwen2.hpp:365-366,388-393` — `kv_half_/kv_q8_`, i8 caches (spike added `kv_tail_/k_tail_/v_tail_`; reverted).
- `src/models/qwen2.cpp` — load alloc + env parse (~:465-478), 3 store sites (:1549 prefill, :1712 decode single-seq, :1929 verify), dispatches (:1155 head, :1325 group), `reset_cache` (:2093); new helper `tail_slot()` (reverted).
- `src/cpu/avx2/attention_avx2.cpp` — head score/value + group score/2 value arms (reverted); `src/cpu/scalar/attention_scalar.cpp` — same 4 sites (reverted).
- Validators: scalar + AVX2 head-arg checks (tail needs q8; reverted).

## 4. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-1395-tail` (no commit ever made on it).
2. Headers + 3 store hooks + 2 dispatches + 8 kernel branches + 2 validator checks, all `spike-1395` marked, env-gated (`TGE_KV_TAIL`), default-off; fixed one self-inflicted build break (dropped q8 block check, restored; then a scalar `exact_tail` redeclaration, renamed).
3. `cmake --build --preset release` clean; `tge_tests` **141/141** (same count as master baseline).
4. `tge fidelity --model Llama-3.2-3B-Q4_K_M --corpus docs/architecture.md --context 512 --tokens 200 --threads 1 --kv i8 --gpu off` (199 positions): flat **KL 2.133e-4 / top-1 196/199 / 200 s**; `TGE_KV_TAIL=64` → **KL 2.087e-5 / top-1 199/199 / 290 s** (10.2× KL win; oracle-load stderr warning is the f32 oracle skipping the tier by design).
5. Thinking-set gate: fixed short prompt ids identical flat-vs-tail (fox loop); fixed-abstract excerpt (architecture.md head, `--n 16 --print-ids`) DIVERGES from token 8 (flat `...83 713 14 55374...` vs tail `...83 713 5640...`) — expected for lossy KV (KIVI diverged at 9; VNNI precedent diverges), teacher-forced agrees 199/199.
6. Warmed reverse-order best-of-3 decode, ~1441-ctx prompt, default threads: TAIL64 **6.89/8.03/8.50** vs OFF **7.07/8.03/7.80** → best 8.50 vs 8.03 (**+5.8%**, inside box noise → parity); attn-phase ms 373-461 vs 402-458 (indistinguishable). First unwarmed round discarded (cold-start 4.70).
7. `python aikit.py check aegean-tge` → 2/4 (test + secrets pass; hygiene + guidelines fail **pre-existing on master**, same as the KIVI-cycle record — README tree-vs-`reports` drift, not this cycle's to fix).
8. Reverted all 5 files (`git checkout -- .`), deleted branch, rebuilt master clean, `tge_tests` 141/141, `git status` clean.

## 5. Measured numbers

| metric (Llama-3B Q4_K_M) | flat i8 | i8 + tail-64 | delta |
|---|---|---|---|
| fidelity mean KL, ctx512/199pos | 2.133e-4 | **2.087e-5** | 10.2× better |
| fidelity worst KL | 1.362e-3 | 3.287e-4 | 4.1× |
| max \|logit diff\| | 2.510e-1 | 1.015e-1 | 2.5× |
| teacher-forced top-1 | 196/199 | **199/199** | +3 |
| decode best-of-3, ~1441 ctx | 8.03 t/s | 8.50 t/s | +5.8% (noise → parity) |
| decode attn phase | 402-458 ms | 373-461 ms | indistinguishable |
| extra resident | — | ~14.7 MB (28L×64×1024×4B×2) | +1.6% of KV @1441 ctx |

Box variance note: unwarmed runs swung 4.70-6.88 for the SAME config; only warmed interleaved best-of-3 counts (both arms).

## 6. Verdict

Keep-grade on quality-at-parity: the slice is a genuine new point between i8 (cheap, KL 2e-4) and f16 (2× bytes, KL 1.7e-6) — KL 2e-5 at +~1% decode, +15 MB. Worth a proposal iff a use case wants sub-f16 quality without doubling KV bytes (e.g. long-context quality floor). Kills nothing; Hadamard/KVarN half still has no carrier (needs sub-8-bit bulk — same wall as IDEA-1515/1516).

## 7. Reimplementation recipe (keep-grade)

1. `kernels.hpp`: add to both Args structs `const float *tail_keys, *tail_values; int tail_n, tail_first, tail_start;` (defaults 0/nullptr).
2. `qwen2.hpp`: add `int kv_tail_; vector<float> k_tail_, v_tail_;`.
3. `qwen2.cpp`: `tail_slot(seq,layer,pos)` helper as in §3; load: `atoi(getenv("TGE_KV_TAIL"))`, alloc `seq*layers*N*kv_dim` floats ×2 (warn-and-off unless `kv_q8_` — fidelity oracle loads f32 by design); hook all 3 `store_kv_q8` call sites with ring memcpy; dispatches: head (`tail_first = pos+1-N`, `tail_start = 0`, base = `(seq*layers+layer)*N*kv_dim + kv_head_offset`); group (same + `tail_start = start`); `reset_cache` clears tail vectors.
4. Kernels (scalar + AVX2, head + group, score + value): `exact = q8 && tail_n>0 && tail_start+token >= tail_first`; score: `dot_avx2(query, tail_keys + (abs%N)*stride)`; value: float accumulate from `tail_values` row. Validators: tail requires q8 + non-null ptrs.
5. Price exactly as §4-5 (same model/corpus/ctx/tokens/threads; warmed reverse-order best-of-3 for decode). Kill on (a) KL ≥ flat i8 or (b) decode < 0.95× flat i8 best-of-3.

## 8. Citation verification (goal requirement)

- `Anbeeld/beellama.cpp` (GitHub, fork of llama.cpp, **1.2k★ / 82 forks**, MIT): LIVE. About line confirms "variance-normalized KV-cache quantization (KVarN), KV cache precision tail for recent tokens". README confirms KVarN widths kvarn2-8 via `--cache-type-k/v`, `--kv-tail-tokens` recent-F16/BF16, single-global-softmax merge under FlashAttention. Benchmark ladder confirms dossier numbers: Qwen3.6-27B row cited, `kvarn5/kvarn4` = 1360 MiB / 33.2% / median KLD **0.000936** vs dossier "≈0.0009 at 33%" — ALIGN. KVarN origin "Huawei / fork", KVarN for speculative draft caches, Gemma-4 sliding-window caveat all present — dossier summary accurate.
- Dossier delta vs tree: "Add i8 next step after f16" is ALREADY SHIPPED (`--kv i8` kept 2026-11-02, KL 3.2e-4) — the dossier predates/ignores the landed tier; only the tail (+Hadamard) is new work. RECORDED so the next cycle doesn't re-litigate it.
- Internal anchors (§3 line numbers) verified by grep on today's `master` before spiking.

## Commands run and their results

- `git status --short` → clean; `git checkout -b spike-1395-tail` → switched; final `git checkout -- .` + `git checkout master` + `git branch -D spike-1395-tail` → master, clean (verified: `git status` clean, stash list = pre-existing 2026-11-04 peer stash only).
- Webfetch `https://github.com/Anbeeld/beellama.cpp` → 200, README/ladder as quoted in §8.
- Grep `qwen2.cpp/qwen2.hpp/kernels.hpp/attention_*.cpp` → anchors in §3; `Hadamard` → 0 hits (carrier gap confirmed).
- `cmake --build --preset release` → 54/54 clean (after fixing 2 self-inflicted spike compile errors); `tge_tests` → 141/141 (spike) and 141/141 (post-revert master).
- `tge fidelity ... --kv i8` → KL 2.133e-4, 196/199, 200 s; `TGE_KV_TAIL=64` → KL 2.087e-5, 199/199, 290 s.
- `tge generate ... --print-ids` fox prompt → identical; architecture.md excerpt → diverges token 8 (recorded strings in §4.5).
- `tge generate` ~1441-ctx best-of-3 (warmed, reverse-order): numbers in §5.
- `python aikit.py check aegean-tge` → 2/4, hygiene+guidelines fail pre-existing (KIVI-cycle precedent).
