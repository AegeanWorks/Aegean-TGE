# WINNER PORT: --kv auto — LANDED, keep-grade (mapping exact on all arms)

- **Source recipe**: `reports/idea-invented-kv-auto-proceed.md` (flag-first verdict: f16 numerics move bits, so auto stays opt-in, default remains f32).
- **Port commit**: `bfa7583` (`feat(aegean-tge): --kv auto flag selects f16 above 512 context else f32`), 4 files (+69/−7): rule (`auto_kv_half` + `kAutoKvHalfContext = 512`), 4 CLI sites (generate/fidelity/continuous/sessions) with resolve-before-call, threshold unit test. Nothing pushed (human gate).
- **Verdict**: WINNER (mapping win) — kept. The flag routes long-context runs to the priced f16 tier automatically with zero short-context delta.

## 1. What landed

`Qwen2::auto_kv_half(context_limit)`: declared capacity (0 = loader default 4096) above 512 → f16 else f32; i8 never auto. Each of the 4 `--kv` sites accepts `auto`, resolves post-parse to the same `kv_half/kv_q8` bools, updates its error string; reference (fidelity candidate) arm untouched — it resolves auto the same way. Help strings updated. Default unchanged (explicit f32/f16/i8 behave exactly as before).

## 2. Remeasure (this tree; interleaved where timing matters)

- Fidelity (0.5B-Q8_0, arch corpus, 1023 pos, ctx 1024): manual f16 == auto **digit-for-digit** (KL 2.142e-06, worst 2.154e-04, 1022/1023).
- Short (story, ctx 256): auto ids == manual f32 ids byte-identical (`11 1052 572 ...`).
- Long (Llama-3B-Q4_K_M, fox×200, ctx 4096, 62 tok, interleaved f16/auto/f16/auto): f16 7.11/6.57 (mean 6.84), auto 7.06/7.59 (mean 7.33) — same kernels (attn 12%, matvec 86% both arms), 62 ids identical across all runs; absolutes breathe with the box (recipe warned: 13% drift, interleave — done). f32 control 6.39 t/s (attn 18%): auto runs +12-15% over default f32, matching the recipe's +9-16%.
- Thinking set on auto (default ctx → f16): story + haiku ids byte-identical to every prior run.

## 3. Gates

Build 54/54, `tge_tests` 143/143 (incl. new `auto_kv_half_threshold`: 0/1/511/512/513/1024/4096/32768 + constexpr; one spike-test bug fixed: ctx 0 = default 4096 → auto-true). `aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing).

## 4. Why it stays

Every falsifier arm passes with zero behavior change outside the flag: digit-identical fidelity, byte-identical short ids, long runs on the same f16 kernels with identical ids and the expected +12-15% over f32. No default changed, no numerics invented.

## 5. Next winner in queue

kv-auto closes the port shelf's flag-size items. Remaining: no-gather sparse follow-ups (prefill path, f16 weights), 1517 stage-2 follow-ups, 1714 hardening — same treatment.

(End of file.)
