# IDEA-1515: Budget-Aware Mixed-Precision KV Tier (MultiAlloc) — KILLED analytic (no carrier on today's tree)

- **Idea ID**: IDEA-1515 (dossier `scout-filing-2026-aegean-tge.md` §IDEA-1515: MultiAlloc, arXiv 2610.18642 — profiling-guided layer-wise + token-wise precision allocation under a fixed KV budget, policy frozen at inference, no online search; falsifier: profiled Qwen2.5-0.5B policy beats flat i8 at matched bytes with sub-8-bit layers decodable inside AVX2, or decode regresses vs `--kv i8` → kill).
- **Cycle date**: 2026-11-07
- **Verdict**: KILLED analytic — the dossier's own must-be-true already fails on today's `master`. No spike built (loop rule: no spike for dead ideas). Nothing touched; tree exactly as found.
- **Tree state at close**: `git status` clean (no diff), no branch created, no stash change, `master`. Nothing committed except this report. Nothing pushed (human gate).

## 1. Mechanism (as proposed)

Offline calibration profiles per-layer KV sensitivity plus per-token importance, then bakes a static per-layer bit-width table at load: sensitive layers up (f16/fp window for exact tokens), robust layers down (sub-8-bit). At inference the decoder reads each layer's cache in its baked precision — total KV bytes fixed at or under the flat-i8 budget, quality above flat i8 because bits move to where they matter. Target files per dossier: `src/models/qwen2.cpp` (KV store), `src/models/plan.cpp` (budget manager), `include/tge/` headers.

## 2. Predicted gain — with the math that kills it

Shipped KV precisions on today's tree are whole-model flags only: f32 (4 B/elem), f16 (2 B/elem), i8 (≈1.03 B/elem with per-32 scales) — `qwen2.cpp:465-478`, `qwen2.hpp:182-188`. `plan.cpp:66-67` prices KV at f32 (`block_count * kv_dim * 4 * 2`); there is no per-layer table anywhere in `plan.cpp`.

Matched-bytes arithmetic for the reference model (Llama-3.2-3B, 28 layers): flat i8 ≈ 28 layer-equivalents × 1.03 B. Any mixed policy containing an f16 (or fp-window) layer at 2-4 B/elem must offset it with layers BELOW 1.03 B/elem to hold the budget. No sub-8-bit KV decode path ships: the only 2-bit attempt (IDEA-1516/KIVI stage-2a, `idea-1516-kivi-stage2a-killed.md`) died on fidelity — mean KL 1.285e-1 vs the 1e-3 bar (128× over), top-1 75/86. The remaining `nibble` code in `src/cpu/avx2/dequant_avx2.cpp` is weight dequant, not a KV read path.

So on today's tree the achievable "mixed" set is {f16, i8} only, whose average can never go below flat i8 bytes. Predicted gain at matched bytes: ≤ 0 by construction — the best legal table is all-i8, i.e. today's `--kv i8` (decode 7.63 t/s at 2048 ctx, KL 3.2e-4, `measurement-log.md` 2026-11-02). Decode is bandwidth-bound (engine log: prefill/GEMV scale with bytes moved), so a table with any f16 layer moves strictly more bytes/token than flat i8 and cannot beat 7.63 t/s. Both falsifier arms trip without running anything.

## 3. Falsifier (from the dossier)

"Profiled Qwen2.5-0.5B policy beats flat i8 at matched bytes with sub-8-bit layers decodable inside the AVX2 budget", or "decode regresses vs `--kv i8`". Trips analytically: sub-8-bit layers are not decodable (KIVI killed stage-2a on numerics before any AVX2 path), and any f16-containing mix regresses decode bytes/token vs i8. No benchmark needed; a spike would re-prove the KIVI kill at higher cost.

## 4. File anchors (current `master`, untouched)

- `src/models/qwen2.cpp:50` (`kv_dim`), `:68-101` (`quantize_kv_block`, `store_kv_q8`), `:465-478` (whole-model `kv_half_`/`kv_q8_` alloc — the hook a table would need per-layer variants of), `:1139-1167` (decode read dispatch f16/i8/f32 — the site needing a 4th sub-8-bit arm that does not exist).
- `include/tge/models/qwen2.hpp:119-120` (`load(..., kv_half, kv_q8)` signature), `:182-188` (tier predicates), `:365-366` (tier flags), `:388-411` (i8 cache + scale members).
- `src/models/plan.cpp:65-69` (f32 KV pricing), `:91-124` (`place()` — session/context shrink only, no precision policy).
- Precedents: `reports/idea-1516-kivi-stage2a-killed.md` (no sub-8-bit carrier), `docs/measurement-log.md` 2026-11-02 (`--kv i8` kept: +32% vs f32, KL 3.2e-4) and 2026-11-01 (`--vnni` KL 1.568e-3 price tag convention).

## 5. What was done (no spike)

1. `git status --short` clean, `master`, log head `45771cd` — recorded baseline.
2. Verified dossier source live: `https://arxiv.org/abs/2610.18642` resolves — MultiAlloc, Zhou et al., submitted 13 Oct 2026, cs.LG. Abstract confirms the dossier's summary nearly verbatim: profiling-guided allocation between layer-wise enhancement and token-wise preservation under fixed budget, no online search, beats layer-wise/token-wise/multi-granularity baselines at matched budgets on Mistral/Llama/DeepSeek-Qwen/QwQ. Source↔dossier alignment GOOD (dossier "Qwen families" = paper's "DeepSeek-Qwen, and QwQ").
3. Grepped current tree for carriers: no per-layer table, no sub-8-bit KV read path, no mixed policy — analysis above.
4. No branch, no build, no bench: nothing to measure because the must-be-true fails. (Gates that count benchmarks do not trigger when no benchmark counts.)

## 6. Measured numbers

None this cycle — analytic kill. Reference numbers (pre-existing, not re-run): flat i8 decode 7.63 t/s @2048 (vs 5.80 f32, +15% vs f16); i8 KL 3.2e-4; KIVI-2-bit KL 1.285e-1 (128× over bar).

## 7. Thinking-set verdict

N/A — no numerics changed, no behavioral surface changed. (Fixed abstract + creative set only gates spikes that touch numerics or sampling.)

## 8. Reimplementation recipe (keep-grade: KILLED — rebuild only if the carrier lands)

Worth a future only after a sub-8-bit KV tier ships with KL ≤ 1e-3 (i.e. what KIVI stage-2a failed). Then, and only then:

1. `include/tge/models/qwen2.hpp`: replace `kv_half_`/`kv_q8_` bools with `std::vector<int> kv_bits_per_layer_` (values 16/8/4/2) + `int kv_fp_tail_tokens_`; extend `load()` signature; add per-layer cache member variants (or a tagged union of per-layer buffers + scales).
2. `src/models/qwen2.cpp`: in `load`, allocate per layer by bits (mirror `store_kv_q8` blocking for i8 layers; new packed store for sub-8-bit); extend the three read sites (prefill/verify/decode `else` chains at the `kv_half_`/`kv_q8_` dispatch, cf. `:1151-1167` decode) with per-layer arms; add offline profiler (calibration prompt set → per-layer sensitivity = KL vs fp when that layer alone is quantized) that emits the frozen table; gate behind `--kv mixed:<table>` with fail-loud on unknown layers.
3. `src/models/plan.cpp`: replace scalar `kv_bytes_per_token` with `sum_layer_bits/8 * 2 (K+V)`; `place()` unchanged otherwise.
4. Price like precedents: `tge fidelity` mean KL + top-1 vs flat i8 at matched bytes (bar: KL ≤ flat-i8 KL, top-1 ≥ flat-i8), then `tge bench` decode best-of-3 vs 7.63 t/s baseline. Kill if either arm fails — same falsifier.
5. Constants from this analysis: keep per-32 scales for i8 layers; fp tail = last-N tokens exact (N from profiler, cf. IDEA-1395's kv-tail-tokens); never share rows across precisions (cf. S6 kind-tag lesson in measurement-log).

## 9. Citation verification (goal requirement)

- `arXiv:2610.18642` (MultiAlloc, cs.LG, v1 13 Oct 2026, Zhou et al., 8 authors): LIVE, abstract matches dossier claim (fixed-budget layer+token allocation, offline profiling, no online search, beats uniform baselines incl. Mistral/Llama/Qwen-family at matched budgets). ALIGN: yes.
- `reports/idea-1516-kivi-stage2a-killed.md` + `docs/measurement-log.md` (2026-11-01/02 i8/VNNI/f16 prices): internal, quoted numbers match the log rows cited. ALIGN: yes.

## Commands run and results

- `git status --short` → clean (empty); `git log --oneline -3` → 45771cd / e3ca9dd / ea8c747; `git branch --show-current` → master.
- Webfetch `https://arxiv.org/abs/2610.18642` → 200, abstract as quoted in §5.2.
- Grep `qwen2.cpp`/`qwen2.hpp`/`plan.cpp` for KV/tier/budget symbols → anchors in §4; no per-layer or sub-8-bit KV hits (nibble hits are weight dequant).
- No build/bench (analytic kill; tree untouched — verified by clean status at close).
