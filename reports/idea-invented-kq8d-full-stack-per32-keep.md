# INVENTED: full-stack kq8d under per-32 activation scales (`head,ffn,down,attn`) — KEEP (393/399, KL 4.382e-4, ~1.42x decode)

- **Idea**: The per-256 full stack (`head,down,attn`) was killed at 1.576e-3 / 97.33% because summed activation error crossed the class bar. Per-32 activation scales cut that error ~2.5x per scope and rescued the FFN stack (392/399). Extend the same per-32 layout to the last convertable share — the attention projections q/k/v/o — and price one tier that runs **91% of the per-token weight stream through the integer row kernel**: `--kq8d head,ffn,down,attn`.
- **Cycle date**: 2026-11-11
- **Verdict**: **KEEP (report-only; integration is a human-gated decision).**
  - Fidelity at 399 positions, f32 KV: **mean KL 4.382e-4**, worst 5.774e-3, **top-1 393/399 (98.50%)** — inside the 1.6e-3 / >= 97.6% class bar. For comparison, the per-256 full stack was 1.576e-3 / 97.33% (failed) and the per-32 max tier (no attn) is 2.749e-4 / 394/399.
  - Decode (rotated, n=48, threads 10): max tier `head,ffn,down` **1.326/1.282/1.311** (median ~1.31x); full stack **1.346/1.422/1.445** (median **~1.42x**, 3/3 above the max tier); incremental full/max 1.015/1.109/1.102.
  - Ids: the full stack flips the same single token the `attn`/`head` scopes flip on this prompt (`A1-62-BF` vs `6B-10-8A`); expected for a lossy tier at this KL.
- **Tree state at close**: spike reverted from `src/cpu/avx2/dequant_batch_kq8_avx2.cpp`, `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp`, `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

Builds directly on the two per-32 reports (`idea-invented-kq8d-per32-activation-keep.md`, `...-per32-q6k-max-tier-keep.md`):

- The single-row kq8d kernel already quantises activations per 32 values for both K-quants; `--kq8d attn` already routes `wq/wk/wv/wo` through it (`kq8d_attn_`), and the FFN scope routes gate/up/down. The only missing piece was a scope that enables **all four** groups at once.
- Spike adds the CLI scope `head,ffn,down,attn` (both `generate` and `fidelity`): sets `kq8d_head`, `kq8d_gateup`, `kq8d_down`, `kq8d_attn`.
- Per-token coverage: head (15% of weights, Q6_K) + gate/up (37%, Q4_K) + down (19%, Q4_K) + q/k/v/o (19%, mixed Q4_K/Q6_K — `attn_v` is Q6_K in Q4_K_M, so both per-32 dots are exercised) = ~90-91%. Only the embedding lookup and the norms stay on the float path.
- Prefill/batch (`gemv_dequant_batch_kq8_avx2`, per-256 Q8_K) is untouched; this is a decode-only tier.

## 2. Predicted gain (with the math)

Coverage 0.91 at an effective ~1.5x on the converted share predicts `1/(0.09 + 0.91/1.5) = 1.42x`; at 1.7x effective, 1.52x. Measured 1.35-1.45x (median 1.42) sits exactly on the 1.5x-effective line, consistent with the max tier's 1.31x at 0.71 coverage. `attn` adds 19 points of coverage, so the incremental prediction is ~1.08-1.10x; measured 1.02-1.11x.

## 3. Falsifier (stated before the spike)

**KEEP** if the full stack at 399 positions has mean KL <= 1.6e-3 and top-1 >= 390/399 **and** decodes >= 1.10x over the max tier (or >= 1.35x over off) in rotated pairs. **KILL** at the first failing gate.

Measured: 4.382e-4 / 393/399; 1.35-1.45x over off and 1.02-1.11x over max. Keep.

## 4. File anchors (current HEAD, read-before-write verified)

- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:397-456` — `gemv_dequant_kq8_row_avx2` (single-row entry; the per-32 rewiring target).
- `:21-27` — `Q8KBlock`; `:63-98` — `quantize_row_q8k`; `:143-216` — `dot_q4k_q8k`; `:222-313` — `dot_q6k_q8k`; `:112-136` — `k_scale_shuffle`.
- `src/models/qwen2.cpp:860-861` (head), `:2178-2180` (q/k/v, `kq8d_attn_`), `:2218-2219` (o, `kq8d_attn_`), `:2228-2229` (gate/up), `:2233-2234` (down).
- `include/tge/models/qwen2.hpp:241-259` — `set_kq8d(head, down, attn, gateup)`.
- `apps/cli/main.cpp:2286-2310`/`:2805-2830` — `--kq8d` parsers; `:506`/`:2486` — signatures; `:591`/`:2549` — `set_kq8d` calls.
- `reports/idea-invented-kq8d-full-stack-killed.md` (per-256 full stack 1.576e-3 / 97.33%, superseded here), `reports/idea-invented-kq8d-per32-activation-keep.md`, `reports/idea-invented-kq8d-per32-q6k-max-tier-keep.md` (the two halves this composes).

## 5. What was done (temporary spike, fully reverted)

1. Re-applied the per-32 kernel from the two prior recipes: `Q8K32Block { d[8], qs[256], sums32[8], sums16[16] }`, `quantize_row_q8k32`, `unpack_q4k_scale_min`, `dot_q4k_q8k32`, `dot_q6k_q8k32`; row kernel uses one plain local `std::vector<Q8K32Block>` and branches on kind.
2. Re-applied the `ffn` wiring (`set_kq8d` fourth flag, gate/up matvec flags) and the `ffn`, `ffn,down`, `head,ffn,down` parse cases.
3. Added `head,ffn,down,attn` to both parsers (sets all four scope flags).
4. Built; 171/171. Fidelity: full stack at 399 positions. Speed: three rotated three-arm pairs (off / max / full), n=48, threads 10, `--print-ids`.
5. Reverted four files; rebuilt; 171/171; gates 4/4; clean tree.

## 6. Measured numbers

### Fidelity (f32 KV, scalar oracle, docs/architecture.md, 399 positions, context 1024)

| scope | layout | mean KL | worst KL | top-1 | bar |
|---|---|---|---:|---:|---:|---|
| head,down,attn (per-256, killed) | per-256 | 1.576e-3 | — | 97.33% | **fail** |
| head,ffn,down (per-32) | per-32 | 2.749e-4 | 2.735e-3 | 394/399 (98.75%) | pass |
| **head,ffn,down,attn** | **per-32** | **4.382e-4** | **5.774e-3** | **393/399 (98.50%)** | pass |

The full stack costs 1 top-1 flip vs the max tier and roughly doubles its KL, but stays well inside the class bar — the per-32 layout is what makes the last 19 points of coverage affordable.

### Decode (dragon prompt, n=48, f32, threads 10, rotated arm order)

| pair | off | head,ffn,down | head,ffn,down,attn | max/off | full/off |
|---:|---:|---:|---:|---:|---:|
| 1 | 17.13 | 22.72 | 23.05 | 1.326 | 1.346 |
| 2 | 17.36 | 22.25 | 24.68 | 1.282 | 1.422 |
| 3 | 17.06 | 22.37 | 24.66 | 1.311 | 1.445 |
| **median** | 17.13 | 22.72 | 24.66 | **1.311** | **1.422** |

Incremental full/max: 1.015, 1.109, 1.102 (median ~1.10x).

### Ids (greedy, `--print-ids`)

| arm | ids hash |
|---|---|
| off | `6B-10-8A` |
| head,ffn,down | `6B-10-8A` |
| head,ffn,down,attn | `A1-62-BF` |

The flip enters with the `attn` scope (same token the standalone `attn`/`head` scopes flip on this prompt). The 98.50% top-1 at 399 positions is the honest statement; matching ids is not guaranteed.

## 7. Integration consequences

- This is the maximal kq8d decode tier: ~91% of weights integer at ~1.42x, KL 4.382e-4 / 98.50% on f32 KV. It supersedes `head,ffn,down` as the headline opt-in tier only for users who accept the extra flip; `head,ffn,down` (98.75%, 1.31x) remains the safer pick.
- All scopes now share the per-32 layout; the shipped per-256 pricings for `head`, `down`, `head,down`, `attn` are superseded and must be re-stated at 399 positions before any default/flag documentation changes.
- Stacking with i8/f16 KV and with speculation is still unpriced here. The spec caveat stands: the verifier uses float-only batch math while the corrected forward would be integer per-32, so near ties can flip accepted drafts; do not enable both without an id check.
- Prefill numerics are unchanged (batch kernel untouched); only decode is affected.

## 8. Full reimplementation recipe

1. Apply the per-32 kernel from `reports/idea-invented-kq8d-per32-q6k-max-tier-keep.md` §8 steps 1-4 (Q8K32Block with `d[8]/qs[256]/sums32[8]/sums16[16]`, `quantize_row_q8k32`, `unpack_q4k_scale_min`, `dot_q4k_q8k32`, `dot_q6k_q8k32`, row-kernel branch with a plain local buffer).
2. Apply the `ffn` wiring: `set_kq8d(head, down, attn = false, gateup = false)` + `kq8d_gateup_`; gate/up matvec calls pass `kq8d_gateup_`; CLI threads `kq8d_gateup` through both commands and both `set_kq8d` calls.
3. In both `--kq8d` parsers add the combined cases: `ffn` (gateup), `ffn,down`/`down,ffn` (gateup+down), `head,ffn,down` (head+gateup+down), and **`head,ffn,down,attn`** (all four flags). Reset all four flags before the comparison chain; extend the error text and both usage blocks to list the new scopes.
4. Build + `tge_tests` 171/171. Fidelity: `tge fidelity --kq8d head,ffn,down,attn --context 1024 --tokens 400 --threads 1 --gpu off` -> expect ~4.4e-4, 393/399.
5. Speed: rotated three-arm pairs of `generate --n 48 --kv f32 --threads 10 --gpu off [--kq8d head,ffn,down|head,ffn,down,attn]` -> expect max ~1.31x, full ~1.42x, full/max ~1.10x.
6. Revert (report-only) or ship behind the opt-in flag after human approval, re-pricing the other scopes under per-32 first.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge fidelity ... --kq8d head,ffn,down,attn --context 1024 --tokens 400 --threads 1 \
  --gpu off                                                        -> 4.382e-4, 393/399
tge generate ... --n 48 --kv f32 --threads 10 --gpu off --print-ids \
  [--kq8d head,ffn,down|head,ffn,down,attn]                        -> §6 speed table
git checkout -- src/cpu/avx2/dequant_batch_kq8_avx2.cpp include/tge/models/qwen2.hpp \
  src/models/qwen2.cpp apps/cli/main.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the rebuilt binary; the committed tree is unchanged except this report.
