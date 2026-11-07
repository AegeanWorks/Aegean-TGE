# IDEA-1395 tail as a real flag (`--kv-tail-tokens`) — MEASURED through the flag, then REVERTED per instruction

- **Idea ID**: IDEA-1395 tail-slice landed as a product flag, then reverted (not kept). Validation of the exact code that would ship: `Qwen2::load(..., kv_tail)` plumbing, `--kv-tail-tokens N` on `generate` + `fidelity`, hermetic kernel test, usage text.
- **Cycle date**: 2026-11-07
- **Verdict**: MEASURED through the flag, then fully reverted per explicit instruction. Flag-path numbers reproduce the spike bit-exactly (same KL to 4 significant digits, same worst-case, same ids strings). Nothing of the feature remains on any branch.
- **Tree state at close**: `master`, `git status` clean, `feat/kv-tail` branch deleted (2 commits discarded: `bb26304` feat + `2b36fec` docs), stash untouched (pre-existing peer stash 2026-11-04 only). Final gates on master: build clean, `tge_tests` 141/141. Nothing pushed (human gate).

## 1. What was landed (then reverted)

Same mechanism as `idea-1395-tail-slice-measured.md` §7, but as product code instead of env-gated spike: `kv_tail` as the last `Qwen2::load` parameter (default 0, all existing callers compile unchanged); `--kv-tail-tokens N` on `generate` + `fidelity` with CLI validation (tail needs `--kv i8`, rejects negative, load rejects tail > context); ring store at all 3 store sites + `reset_cache` clear; head + GQA-chunk dispatch; 8 kernel branch sites (scalar + AVX2, head + group, score + value) with fail-loud validators; usage text for both commands. Hermetic test `attention_head_tail_matches_float_and_scalar` committed with the feature: full coverage bit-equals scalar float, partial tracks scalar-vs-AVX2 at 1e-5, tail-without-q8 and null buffers throw on both ISAs.

## 2. Measured numbers (flag path, Llama-3B-Q4_K_M)

| check | flag result | spike result (Cycles 2-4) | match |
|---|---|---|---|
| fidelity arch/ctx512/200tok/1T, N=64 | KL **2.087e-5**, worst 3.287e-4, top-1 **199/199** | identical to 4 s.f. | exact |
| fidelity validation | `--kv f32 --kv-tail-tokens 64` → clean `needs --kv i8` error | n/a (env silently skipped) | stricter, as designed |
| decode best-of-3 warmed reverse-order ~1441 ctx, N=64 | 8.00 (7.92/7.89/8.00) vs flat 8.73 (8.73/7.97/8.44) = −8.4%, inside ±20% box → parity | +5.8% / +0.7% earlier rounds | same verdict |
| fox-loop ids | `578 39935 35308 13` ×4, flat-identical | identical | exact |
| abstract-excerpt ids | `683 374 7559 520 1595 83 713 5640 713 14 63 ...` | identical string to spike | exact |
| `tge_tests` | **142/142** (141 + new tail test) | 141/141 (spike era) | green |
| aikit | 3/4 (test+secrets+guidelines pass; hygiene inconclusive-fail is structural — C++ repo, nothing scannable; gaming it with dummy files is forbidden) | 2/4 (guidelines failed on README `reports` drift, fixed in this work) | improved |

Three decode rounds now read +5.8 / +0.7 / −8.4 — all inside box noise, parity verdict unchanged. Attn-phase ms indistinguishable every round.

## 3. Falsifier (stated for the landing)

Flag path diverges from spike numbers (would indicate a porting bug), or decode regresses > 5% consistently, or the hermetic test fails. None tripped: flag output is spike output to the printed digit, including the abstract-excerpt id string.

## 4. File anchors (as landed, now reverted — for the record)

- `include/tge/models/qwen2.hpp`: `load(..., int sequences = 1, int kv_tail = 0)` + `kv_tail_/k_tail_/v_tail_` members.
- `include/tge/cpu/kernels.hpp`: `tail_n/tail_first/tail_start/tail_keys/tail_values` on both Args structs.
- `src/models/qwen2.cpp`: `tail_slot()` helper, load validation + alloc, 3 store hooks, 2 dispatches, `reset_cache` clear.
- `src/cpu/avx2/attention_avx2.cpp`, `src/cpu/scalar/attention_scalar.cpp`: 8 branch sites + 2 validators.
- `apps/cli/main.cpp`: `--kv-tail-tokens` parse + validation on `generate`/`fidelity`, pass-through to `load`, usage text ×2. (Bench paths intentionally left default-off.)
- `tests/test_attention.cpp`: `attention_head_tail_matches_float_and_scalar`.
- Docs (reverted with the branch): measurement-log §2026-11-07 entry, README tree `reports/` line.

## 5. Reimplementation recipe

`idea-1395-tail-slice-measured.md` §7 (kernel/store recipe, re-applied 4× first-try clean) + §1 above for the product surface (load param order, CLI flag name/validation, hermetic test shape). To rebuild exactly: apply the 5-file spike recipe, then the flag/test/usage deltas in §4's file list.

## 6. Revert record

`git checkout master` (report file, untracked, survives the switch) → `git branch -D feat/kv-tail` (deleted `bb26304` + `2b36fec`; nothing pushed, nothing merged, nothing cherry-picked elsewhere) → `git status` clean → rebuild clean → `tge_tests` 141/141. Verified: `git branch -vv` shows only `master`; `git stash list` unchanged (peer stash only).

## Commands run and their results

- Feature build (7 code files + 1 test): `cmake --build --preset release` → clean; `tge_tests` → 142/142.
- `tge fidelity --kv i8 --kv-tail-tokens 64` (arch/ctx512/200/1T) → KL 2.087e-5, 199/199; `--kv f32` + tail → clean rejection.
- `tge generate` warmed reverse-order best-of-3 (~1441 ctx) → §2 numbers; `--print-ids` fox + abstract → §2 strings.
- `python aikit.py check aegean-tge` → 3/4 (hygiene structural).
- Revert + `git status` (clean) + rebuild + `tge_tests` 141/141.
