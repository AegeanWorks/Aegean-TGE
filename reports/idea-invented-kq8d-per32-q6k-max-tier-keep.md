# INVENTED: per-32 activation scales for Q6_K + the `head,ffn,down` max tier — KEEP (394/399, KL 2.75e-4, ~1.38x decode)

- **Idea**: Extend last cycle's per-32 Q8 activation layout from Q4_K to Q6_K, so the LM head gets the same finer activation quantisation, and combine it with the whole-FFN per-32 stack into one max tier: `--kq8d head,ffn,down` (head + gate/up + down through the integer row kernel). Predicted: the head's fidelity improves several-fold (its per-256 KL was 1.874e-4), the combined stack stays inside the 1.6e-3 class bar, and decode rises toward 1.4x (head 15% of weights + FFN 56%, each at ~1.3-1.5x on its share).
- **Cycle date**: 2026-11-11
- **Verdict**: **KEEP (report-only; integration is a human-gated decision).**
  - `--kq8d head` alone with per-32: **mean KL 4.557e-5** (was 1.874e-4 at per-256, 4.1x better), worst 1.541e-4, **top-1 398/399 (99.75%)**.
  - `--kq8d head,ffn,down`: **mean KL 2.749e-4**, worst 2.735e-3, **top-1 394/399 (98.75%)** — comfortably inside the 1.6e-3 / >= 97.6% class bar.
  - Decode (rotated, n=48, threads 10): head/off **1.047/1.105/1.072** (median ~1.07), `ffn,down`/off **1.218/1.417/1.262** (median ~1.26), **max tier/off 1.304/1.378/1.401 (median ~1.38x, 3/3)**.
  - Ids: head alone flips one token on the test prompt (`A1-62-BF` vs `6B-10-8A`); the max tier's ids **match off** on this prompt (`6B-10-8A`) — luck on a near tie, not a guarantee.
- **Tree state at close**: spike reverted from `src/cpu/avx2/dequant_batch_kq8_avx2.cpp`, `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp`, `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

Extends `reports/idea-invented-kq8d-per32-activation-keep.md` (Q4_K per-32) to Q6_K:

- `Q8K32Block` grows `std::int16_t sums16[16]` beside `d[8]`, `qs[256]`, `sums32[8]`; `quantize_row_q8k32` fills per-16 sums while quantising each 32-value chunk (same signed-max scheme: `iscale = -127/max`, `d = 1/iscale`).
- `dot_q6k_q8k32(x, y, blocks)`: mirrors the stock Q6_K AVX2 decode (nibbles from `ql`, two high bits from `qh`, minus-32 offset) and the stock scale-shuffle rows; each 32-value chunk `g` gets `k_scale_shuffle(base + k)` (scales `2g` and `2g+1` for its two 16-value halves), `madd_epi16` applies them, and the chunk's activation scale multiplies the converted int32 vector. The −32 offset is corrected per 32-chunk from `sums16[2g]`/`sums16[2g+1]`; the stock kernel's per-block `q8sclsub` path is replaced by that scalar correction.
- `dot_q4k_q8k32` unchanged from the previous cycle (min term now reads `sums32[g]`, which the quantiser still fills).
- Row kernel: both kinds now use the per-32 layout (one plain local `std::vector<Q8K32Block>`; never a `thread_local` captured into the pool lambda — that mistake cost a crash last cycle).
- CLI: `--kq8d head,ffn,down` sets head + gateup + down (both `generate` and `fidelity`); scope list becomes `head|down|attn|ffn|ffn,down|head,ffn,down|head,down`.

## 2. Predicted gain (with the math)

Weight shares: head 15%, gate+up 37%, down 19% -> max tier covers 71% of the stream. If each share converts at ~1.4x effective: `1/(0.29 + 0.71/1.4) = 1.25x`; at 1.5x: 1.32x; measured 1.38x implies an effective ~1.7x on the converted share (the integer row kernel's model-level advantage is larger than the isolated kernel ratios suggest — the same effect seen in the tier re-pricing). Head alone at 15% share and ~1.5x predicts 1.06x; measured 1.07x.

## 3. Falsifier (stated before the spike)

**KEEP** if `--kq8d head,ffn,down` at 399 positions has mean KL <= 1.6e-3, top-1 >= 390/399, and decodes >= 1.15x over the default in rotated pairs; **KILL** at the first failing gate. Also required: the per-32 Q6_K dot passes the existing random-input differential (5% row tolerance) and `head` alone does not lose vs its per-256 fidelity.

All passed: 2.749e-4, 394/399, 1.30-1.40x; head fidelity 4.1x better.

## 4. File anchors (current HEAD, read-before-write verified)

- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:21-27` — `Q8KBlock`; `:63-98` — `quantize_row_q8k`; `:143-216` — `dot_q4k_q8k`; `:222-313` — `dot_q6k_q8k`; `:397-456` — `gemv_dequant_kq8_row_avx2` (the branch the spike rewires); `:112-136` — `k4_shuffle_mask` / `k_scale_shuffle` tables (reused by the Q6_K per-32 dot).
- `src/models/qwen2.cpp:860-861` (head), `:2228-2229` (gate/up), `:2233-2234` (down) — kq8d call sites.
- `include/tge/models/qwen2.hpp:241-259` — `set_kq8d` scopes; `apps/cli/main.cpp:2286-2310`/`:2805-2830` — `--kq8d` parsers.
- `reports/idea-invented-kq8d-per32-activation-keep.md` — Q4_K per-32 recipe + the `ffn,down` 6.015e-4 / 392/399 result this cycle builds on.
- `reports/idea-invented-kq8d-lmhead-keep.md` — per-256 head pricing (1.874e-4 / 393-399) superseded for the per-32 layout.

## 5. What was done (temporary spike, fully reverted)

1. Re-applied the Q4_K per-32 kernel and the `ffn`/`ffn,down` wiring from the previous cycle's recipe.
2. Extended `Q8K32Block` with `sums16[16]`, filled it in `quantize_row_q8k32`, kept `sums32[8]`.
3. Added `unpack_q4k_scale_min` (from the Q4_K dot) and `dot_q6k_q8k32`; rewired the row kernel to per-32 for both kinds with a plain local buffer.
4. Added the `head,ffn,down` parse case to both parsers; built; 171/171.
5. Fidelity: `head` and `head,ffn,down` at 399 positions. Speed: three rotated four-arm pairs (off, head, ffn,down, head,ffn,down), n=48, threads 10, `--print-ids`.
6. Reverted four files; rebuilt; 171/171; gates 4/4; clean tree.

## 6. Measured numbers

### Fidelity (f32 KV, scalar oracle, docs/architecture.md, 399 positions, context 1024)

| scope | layout | mean KL | worst KL | top-1 | bar |
|---|---|---|---:|---:|---:|---|
| head | per-256 (prior) | 1.874e-4 | 9.005e-4 (399) | 393/399 (98.50%) | pass |
| **head** | **per-32** | **4.557e-5** | **1.541e-4** | **398/399 (99.75%)** | pass |
| ffn,down | per-32 Q4_K (prior) | 6.015e-4 | 9.310e-3 | 392/399 (98.25%) | pass |
| **head,ffn,down** | **per-32 both** | **2.749e-4** | **2.735e-3** | **394/399 (98.75%)** | pass |

The max tier is *more* accurate than ffn,down alone at 399 positions (2.75e-4 vs 6.02e-4). That is the opposite of the usual "more scopes = more error" intuition and should be read as an observation, not a law: the head's per-32 error may partially cancel the FFN error on this corpus, and the two measurements use different head arithmetic (float vs per-32 integer). A second corpus would be needed to claim generality.

### Decode (dragon prompt, n=48, f32, threads 10, rotated arm order)

| pair | off | head | ffn,down | head,ffn,down | max/off |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.87 | 16.62 | 19.33 | 20.69 | 1.304 |
| 2 | 14.33 | 15.83 | 20.30 | 19.74 | 1.378 |
| 3 | 14.82 | 15.88 | 18.70 | 20.76 | 1.401 |
| **median** | 14.82 | 15.88 | 19.33 | 20.69 | **1.378** |

head/off median ~1.07x; ffn,down/off median ~1.26x.

### Ids (greedy, `--print-ids`)

| arm | ids hash | note |
|---|---|---|
| off | `6B-10-8A` | reference |
| head | `A1-62-BF` | one near-tie flip |
| ffn,down | `6B-10-8A` | matches off on this prompt |
| head,ffn,down | `6B-10-8A` | matches off on this prompt |

The max tier matching off is a near-tie accident on this prompt, not a property; the priced KL (2.75e-4) is the honest statement. Any speculative run must keep the verifier on the same arithmetic.

### Differential

The existing `dequant_batch_kq8_row_matches_float_oracle` (random activations, 5% row tolerance, rows 3 x cols 1024/3072/8192, Q4_K and Q6_K) passes with both kinds on the per-32 layout — the Q6_K scale-shuffle mapping was verified by that test before the fidelity run.

## 7. Integration consequences

- Switching the row kernel to per-32 for both kinds changes every shipped kq8d scope's numerics: `head` improves 4x in KL and 5/399 in top-1; `down` and `head,down` (Q4_K) improve per the previous cycle. Their priced numbers must be re-stated at 399 positions before shipping; the speed effect is small (head ~1.07x either layout, within noise).
- The max tier `head,ffn,down` (~1.38x, 98.75%) becomes the headline opt-in decode tier. It is lossy; the default stays exact.
- Q6_K per-32 costs the head ~5% kernel speed but the measured head tier is unchanged within noise; the fidelity gain is the reason to take it.
- The per-32 Q4_K/Q6_K activation layout is now shared by all kq8d scopes; the batch/prefill kernel (per-256 Q8_K) is untouched, so prefill numerics are unchanged.

## 8. Full reimplementation recipe

1. Apply `reports/idea-invented-kq8d-per32-activation-keep.md` §8 first (Q8K32Block with `d[8]/qs[256]/sums32[8]`, `quantize_row_q8k32`, `dot_q4k_q8k32`, row-kernel Q4_K branch).
2. Extend `Q8K32Block` with `std::int16_t sums16[16]`; in `quantize_row_q8k32`, after quantising each 16-value half, store `sums16[g*2 + half]` and accumulate `sums32[g]`.
3. Add `dot_q6k_q8k32(const Q6KBlock*, const Q8K32Block*, int blocks)`: copy the stock `dot_q6k_q8k` body; for each of the two 128-value halves load `q4bits1/2` and `q4bitsH`, form `q4_0..q4_3`, load `q8_0..q8_3` from `y[i].qs` sequentially; `p16_k = maddubs(q4_k, q8_k)`; `p32_k = madd(cvtepi8_epi16(shuffle(scales, k_scale_shuffle(4j+k))), p16_k)`; `acc = fmadd(cvt(p32_k), set1(y.d[4j+k] * half(x.d_bits)), acc)`; then for k in 0..3 add `32 * y.d[g]*dw * (x.scales[2g]*sums16[2g] + x.scales[2g+1]*sums16[2g+1])` to `offset`; return `hsum(acc) - offset`.
4. In `gemv_dequant_kq8_row_avx2`, build one plain local `std::vector<Q8K32Block>` and branch on kind between `dot_q4k_q8k32` and `dot_q6k_q8k32`. Do not use a `thread_local`.
5. CLI: apply the `ffn`/`ffn,down` wiring, then add `head,ffn,down` (and optionally `head,ffn`): set the three flags; extend the error text and usage.
6. Build + `tge_tests` 171/171 (the row differential covers the Q6_K mapping). Fidelity: `tge fidelity --kq8d head,ffn,down --context 1024 --tokens 400 --threads 1 --gpu off` -> expect ~2.7e-4, 394/399. Speed: rotated pairs of `generate --n 48 --kv f32 --threads 10 --gpu off [--kq8d head|ffn,down|head,ffn,down]` -> expect 1.07x / 1.26x / 1.38x.
7. Revert (report-only) or ship behind the opt-in flag after human approval, re-pricing `down` and `head,down` under the per-32 layout first.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge fidelity ... --kq8d head          --context 1024 --tokens 400 --threads 1 -> 4.557e-5, 398/399
tge fidelity ... --kq8d head,ffn,down --context 1024 --tokens 400 --threads 1 -> 2.749e-4, 394/399
tge generate ... --n 48 --kv f32 --threads 10 --gpu off --print-ids \
  [--kq8d head|ffn,down|head,ffn,down]                            -> §6 speed table
git checkout -- src/cpu/avx2/dequant_batch_kq8_avx2.cpp include/tge/models/qwen2.hpp \
  src/models/qwen2.cpp apps/cli/main.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the rebuilt binary; the committed tree is unchanged except this report.
