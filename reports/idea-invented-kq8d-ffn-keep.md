# INVENTED (retest): kq8d on the FFN gate+up projections — KEEP as an opt-in tier (+1.20x median decode, KL 2.495e-4, 198/199)

- **Idea**: Reopen the kq8d FFN line that `idea-invented-kq8d-combos-killed.md` closed. That kill (2026-11-07) measured `gate+down` and `up+down` against **down-only** on a tree where (a) the single-row integer kernel still heap-allocated its Q8_K scratch per matvec, and (b) the shipped float baseline was the generic dequantise-then-dot path. Since then the row kernel uses a `thread_local` scratch (no per-call allocation) and the default decode path is the fused Q4_K/Q6_K register dot. The kernel bench on today's tree prices the integer row dot at 2.16-2.39x the *generic* float kernel on the FFN shapes, so re-test `gate+up` alone (the scope the old report never isolated) against today's fused baseline.
- **Cycle date**: 2026-11-11
- **Verdict**: **KEEP as an opt-in decode tier** (report-only; not integrated, default unchanged). Six interleaved pairs on Llama-3.2-3B-Instruct-Q4_K_M: **6/6 positive**, ratios 1.291/1.230/1.098/1.191 (dragon) and 1.155/1.209 (periodic), **median ~1.20x**, greedy ids identical in every pair. Fidelity at 199 positions, f32 KV: **mean KL 2.495e-4**, worst 3.883e-3, **top-1 198/199 (99.50%)** — inside the 1.6e-3 / 97.6% class bar and *cheaper* than the shipped down-only tier (7.954e-4 / 390-399).
- **Tree state at close**: spike reverted from `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp`, `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

Decode computes the SwiGLU FFN as `gate = W_gate·h`, `up = W_up·h`, `silu(gate)*up`, `down`. On the reference model both projection matrices are 8192x3072 Q4_K (13.5 MiB each), together 59% of the layer's weight rows. The spike adds a fourth kq8d scope, `ffn`, that routes exactly those two matvecs through the shipped single-row integer kernel `gemv_dequant_kq8_row_avx2` (Q4_K/Q6_K x Q8_K, activation quantised per 256-value block with per-32 scales), while every other projection keeps the fused float dot.

- `Qwen2::set_kq8d(head, down, attn, gateup)` gains a fourth defaulted parameter and `kq8d_gateup_`; `matvec(layer.w_gate, ..., kq8d_gateup_)` and the same for `w_up`.
- CLI `--kq8d ffn` on `generate` and `fidelity`; parser resets all four flags then sets gateup for `ffn`; usage text lists `head|down|attn|ffn|head,down`.
- Nothing else moves: batch/prefill matvecs, the LM head, attention projections, the down projection and all KV tiers stay on their existing paths.

## 2. Predicted gain (with the math)

Kernel bench on today's tree (`tge bench batch`, best of 5, 1 token):

| shape | generic float | kq8d | kq8d/generic |
|---|---:|---:|---:|
| 8192x3072 Q4_K (gate/up) | 0.809 ms | 0.374 ms | 2.16x |
| 3072x8192 Q4_K (down) | 0.881 ms | 0.368 ms | 2.39x |
| 3072x3072 Q4_K (q/o) | 0.432 ms | 0.257 ms | 1.68x |
| 128256x3072 Q6_K (lm head) | 11.314 ms | 6.058 ms | 1.87x |

The fused float dot is ~1.76x the generic kernel on Q4_K (cycle-19 measurement), so kq8d is ~1.23x the *shipped* dot on gate/up. Gate/up are 16384 of 27648 layer rows (59%); at 1.23x on 59% of the work the model-level prediction is `1/(0.41 + 0.59/1.23) = 1.12x`. A 1.3-1.4x effective ratio predicts 1.16-1.21x. Measured median 1.20x.

Why the old combo kill does not apply: it compared extra integer matrices to **down-only**, not to the fused-float default, and it attributed the loss to the per-call quantiser. That attribution is arithmetically wrong on these shapes — quantising 3072 values into 12 Q8_K blocks costs ~9k ops against a 25M-MAC projection (~0.04%) — and the scaffolding has since changed (thread_local scratch, no per-call allocation). The old numbers were also order-drift dominated (one `up+down` run hit 7.20 t/s against down-only's 6.77).

## 3. Falsifier (stated before the spike)

**KEEP as opt-in** if 199-position f32 mean KL <= 1.6e-3 with top-1 >= 97.6%-equivalent **and** interleaved decode median >= 1.05x over the default fused-float binary with identical greedy ids. **KILL** otherwise (extend the combos-killed record; do not re-open again).

Measured: KL 2.495e-4 / 198/199 and median 1.20x, 6/6 pairs. Keep.

## 4. File anchors (current HEAD, read-before-write verified)

- `include/tge/models/qwen2.hpp:241-259` — `set_kq8d` and scope getters (spike adds `gateup`/`kq8d_gateup_` here and at `:478`).
- `src/models/qwen2.cpp:2228-2229` — decode gate/up matvec calls (pass `kq8d_gateup_`).
- `src/models/qwen2.cpp:967-1051` — `matvec` dispatch: kq8d branch first, fused Q4_K/Q6_K second, generic batch fallback.
- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:397-456` — `gemv_dequant_kq8_row_avx2` (tokens==1, thread_local `Q8KBlock` scratch at `:423`, bands over rows).
- `:63-141` — `quantize_row_q8k` / `Q8KBlock` (4 + 256 + 32 bytes per 256-value block).
- `apps/cli/main.cpp:2292`/`:2807` — `--kq8d` parsers; `:506`/`:2486` — command signatures; `:591`/`:2549` — `set_kq8d` call sites.
- `reports/idea-invented-kq8d-combos-killed.md` — the prior kill this cycle supersedes for the isolated `ffn` scope.
- `reports/idea-invented-kq8d-down-keep.md`, `...-lmhead-keep.md`, `...-attn-keep.md`, `...-full-stack-killed.md` — the fidelity stack.

## 5. What was done (temporary spike, fully reverted)

1. Model: `set_kq8d(head, down, attn = false, gateup = false)`; `kq8d_gateup_` member and getter; gate/up matvec calls pass it.
2. CLI: `kq8d_gateup` through `command_generate`/`command_fidelity`; `ffn` parse case in both; usage strings; reset for the new flag.
3. Built, 171/171. Fidelity: `tge fidelity --kq8d ffn --context 512 --tokens 200 --threads 1 --gpu off` on docs/architecture.md.
4. A/B: same binary, `generate --n 64 --kv f32 --threads 10 --gpu off --print-ids`, alternating off/ffn, dragon prompt (4 pairs) and periodic fox-x40 prompt (2 pairs).
5. Reverted all three files with `git checkout --`; rebuilt; 171/171; gates 4/4; `git status` clean.

## 6. Measured numbers

### Fidelity (199 positions, context 512, f32 KV, scalar oracle)

| scope | mean KL | worst KL | top-1 |
|---|---:|---:|---:|
| default (fused float), 399-position precedent, for scale | 1.020e-11 | 2.330e-10 | 399/399 |
| **ffn (gate+up)** | **2.495e-4** | 3.883e-3 | **198/199 (99.50%)** |
| down (prior, 399 pos) | 7.954e-4 | — | 390/399 (97.74%) |
| head+down (prior) | 9.711e-4 | — | 390-399 |
| head+down+attn (prior, fails) | 1.576e-3 | — | 97.33% |

The isolated gate+up tier is the *cheapest* lossy scope measured so far and still clears the bar.

### Decode A/B (same binary, interleaved off -> ffn)

Dragon prompt, 4 pairs, threads 10, f32, greedy:

| pair | off t/s | ffn t/s | ratio | ids |
|---:|---:|---:|---:|---|
| 1 | 14.69 | 18.96 | 1.291 | identical |
| 2 | 15.77 | 19.40 | 1.230 | identical |
| 3 | 16.75 | 18.39 | 1.098 | identical |
| 4 | 14.35 | 17.09 | 1.191 | identical |

Periodic fox-x40 prompt, 2 pairs:

| pair | off | ffn | ratio | ids |
|---:|---:|---:|---:|---|
| 1 | 14.22 | 16.43 | 1.155 | identical |
| 2 | 12.84 | 15.52 | 1.209 | identical |

Median 1.20x, 6/6 positive. Absolute t/s is load-depressed for both arms (the box ran 12-19 t/s all session); only same-pair ratios are claimed.

### Kernel bench (context)

`tge bench batch --tokens 1 --repeat 5`: gate/up 0.809 -> 0.374 ms (2.16x generic); down 0.881 -> 0.368 (2.39x); q/o 0.432 -> 0.257 (1.68x); LM head 11.314 -> 6.058 (1.87x).

## 7. Usage and caveats

- Opt-in, default off: `--kq8d ffn`. The loss is priced at KL **2.495e-4**, top-1 99.50% on f32 KV. Stacking on i8 KV or on `down` is **not** priced here; the prior stacks suggest the combined error stays inside the class bar, but measure before shipping a combined default.
- Under `--speculative` the verifier uses the float-only batch path while the corrected forward now uses kq8d; near-tie flips can make speculative greedy ids diverge from the kq8d-free run. Same caveat as every kq8d scope (the verifier must use the same arithmetic); verify ids before enabling both.
- The activation quantiser is *not* the bottleneck (0.04% of the projection); sharing one Q8_K quant between gate and up is not worth new API surface on this evidence. The gate+up gain is the integer row kernel itself.

## 8. Full reimplementation recipe

1. `include/tge/models/qwen2.hpp`: change `set_kq8d(bool head, bool down, bool attn = false)` to add `bool gateup = false`, assign `kq8d_gateup_`; add the member (`bool kq8d_gateup_ = false;`) and getter.
2. `src/models/qwen2.cpp` decode FFN: `matvec(layer.w_gate, h_.data(), gate_.data(), q_ffn_ptr, kq8d_gateup_)` and the same for `w_up`.
3. `apps/cli/main.cpp`: add `bool kq8d_gateup = false` to both option blocks; parse `ffn` in both `--kq8d` parsers (reset all four flags first, set gateup for `ffn`, error text `head|down|attn|ffn|head,down|off`); thread it through `command_generate` and `command_fidelity`; pass it to both `set_kq8d` calls; update the two usage blocks (`head|down|attn|ffn|head,down`).
4. Build + `tge_tests` 171/171. Fidelity: `tge fidelity --kq8d ffn --context 512 --tokens 200 --threads 1 --gpu off` on the architecture doc; require mean KL <= 1.6e-3 and top-1 >= 97.6%.
5. A/B: same binary, `generate --n 64 --kv f32 --threads 10 --gpu off --print-ids`, alternating off/ffn, at least 4 pairs; require median >= 1.05x and identical ids. Expected: 1.10-1.30x per pair.
6. Revert (report-only spikes), or ship behind the opt-in flag after human approval (push gate).

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge bench batch --rows 8192 --cols 3072 --tokens 1 --repeat 5 --format q4k --kernel {float|kq8d}  -> §2 table
tge fidelity --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --corpus docs/architecture.md \
  --context 512 --tokens 200 --threads 1 --gpu off --kq8d ffn      -> mean KL 2.495e-4, top-1 198/199
tge generate --model ... --prompt <dragon|periodic> --n 64 --kv f32 --threads 10 \
  --gpu off --print-ids [--kq8d ffn]                               -> §6 tables
git checkout -- include/tge/models/qwen2.hpp src/models/qwen2.cpp apps/cli/main.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the rebuilt binary under `build/release/bin/`; the committed tree is unchanged except this report.
