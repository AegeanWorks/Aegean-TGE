# INVENTED: Phi-3.5-Mini i8 Architecture Test — GENERALIZE (3.8B holds 99.4% = Llama's 99.4% to the flip; size rule is architecture-independent)

- **Idea**: The i8 size ladder (conditional keep: holds ≥3B, degrades sub-3B, driver = model-size redundancy not weight format) was built entirely on Qwen/Llama GQA models. Phi-3.5-mini-instruct-Q4_K_M (3.8B) never ran in the engine — and is structurally distinct (32/32 full MHA heads, 3072 dim, 32064 vocab vs GQA Llama). If i8 degradation is architecture-sensitive, a different attention structure at the same size class should diverge; if size-driven, Phi should land in Llama's class.
- **Cycle date**: 2026-11-06
- **Verdict**: GENERALIZE — Phi-3B-class i8 matches Llama-3B-class i8 to the flip: both 508/511 (99.41%, 3 flips), KL 1.39e-04 vs 1.56e-04 (within 12%). The conditional-keep rule sharpens from "≥3B Qwen/Llama" to "≥3B-class regardless of attention structure (GQA and MHA both hold)". No spike (pure measurement, precedented); no branch, no edits, nothing to revert.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). `master` ahead of `origin/master` by 105 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

New mixed-topic corpus (10 paragraphs: photosynthesis, 1453, risotto, offside, counterpoint, transistors, mycorrhizae, free-indirect, packing, RSA — temp file per ladder precedent, construction recorded here verbatim): `fidelity --corpus <file> --context 512 --tokens 600 --threads 1 --gpu off [--kv i8]` on Phi-3.5-mini-Q4_K_M (f32 sanity + i8) and Llama-3.2-3B-Q4_K_M (f32 sanity + i8 control), same window. Coincidence exploited: both tokenizers produced exactly 511 positions, so counts compare directly.

## 2. Predicted gain — a verdict, not a speedup

Expected under the size rule: Phi-i8 within [Llama-i8 − 2% top-1], KL within 3x. Measured: identical.

## 3. Falsifier outcome — CONFIRM-generalize, both arms

1. **Parity**: Phi i8 508/511 (99.41%) vs Llama i8 508/511 (99.41%) — equal to the flip; KL 1.388e-04 vs 1.559e-04 (1.1x).
2. **Sanity**: both f32 arms 511/511 at the floor (Phi 9.7e-11, Llama 8.9e-12) — setup proven, engine runs Phi correctly end-to-end (story generate fluent, greedy, 6.39 t/s, sample 0.0%).

The narrow reading (rule is Qwen/Llama-specific) fails: MHA-vs-GQA, 32k-vs-128k vocab, different training — none of it moves i8 behavior at matched size. Driver stands: redundancy, not architecture.

## 4. File anchors (current `master`, read but untouched)

- `src/models/qwen2.cpp:388-584` (Phi-3 handling: fused attn_qkv, YaRN tables, attn_factor — the support that made this cycle possible).
- Corpus: temp `mixed10.txt` (deleted per scratch discipline; §1 construction reproduces it).
- Grounds: i8 size-ladder (conditional keep + bars), diverse-i8 (architecture question left open), kv-auto (tier rule this sharpens).

## 5. What was done (pure measurement — no spike, no branch)

1. ANALYZE: confirmed the engine loads and runs Phi-3.5-mini (smoke: 32 layers, 3072 dim, 32/32 heads, eager vocab 706 ms, first generate clean) — a first for the loop.
2. Gates: `cmake --build --preset release` 54/54; `tge_tests.exe` 141/141; `aikit.py check` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing, same pair as every prior cycle).
3. Four fidelity runs, same window, back-to-back (§6).
4. One Phi story generate (end-to-end arch check, fluent ids, recorded §3).
5. Temp corpus deleted; no thinking-set diff owed (fidelity verdicts only, precedented); `git status` clean, stash untouched.

## 6. Measured numbers (mixed-topic corpus, ctx 512, 511 positions, threads 1)

| model | KV | KL mean | worst | top-1 | flips |
|---|---|---|---|---|---|
| Phi-3.5-mini-Q4_K_M f32 | f32 | 9.719e-11 | 2.574e-08 | 511/511 | 0 |
| Phi-3.5-mini-Q4_K_M i8 | i8 | 1.388e-04 | 2.253e-03 | 508/511 (99.41%) | 3 |
| Llama-3.2-3B f32 | f32 | 8.935e-12 | 1.288e-10 | 511/511 | 0 |
| Llama-3.2-3B i8 | i8 | 1.559e-04 | 1.377e-03 | 508/511 (99.41%) | 3 |

Run walls: Phi 746/769 s, Llama 734/621 s (fidelity is single-token looped; absolute times unclaimed).

## 7. Full reimplementation recipe (verification protocol — re-run to re-verify)

1. Rebuild the corpus per §1 (10 topics, ~70 tokens each, any mixed long-form works — the gate is comparative, not absolute).
2. `fidelity --model <Phi-3.5-mini|reference> --corpus <file> --context 512 --tokens 600 --threads 1 --gpu off [--kv i8]`; expect f32 floors + i8 parity across architectures at matched size (top-1 within ~1%, KL within ~3x).
3. Revival trigger: a ≥3B model whose i8 diverges from its size class (flips >> class rate or KL >> 3x) — that reopens architecture-dependence with a named counterexample, and narrows the keep rule to the architectures that hold.

## 8. Verdict and why

**GENERALIZE: the i8 size rule is about redundancy, confirmed across attention structures.** Phi-3.5-mini matches Llama-3B to the flip on identical footing — the strongest cross-architecture confirmation the loop has (different heads, vocab, training, same outcome). The kv-auto tier rule now reads architecture-free at the top end: ≥3B-class → i8 stands; sub-3B → f32/f16. Side value: the engine's Phi-3 support is proven end-to-end for the first time (load, prefill, decode, fidelity, story). Next-cycle candidates: the 1517 stage-2 skip harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- Phi smoke: `generate --model <Phi-3.5-mini-Q4_K_M> ... --n 4` → clean (32 layers, 3072 dim, 32/32 heads).
- `cmake --build --preset release` → 54/54; `tge_tests.exe` → 141/141; `aikit.py check aegean-tge` → 2/4 (test + secrets pass; pre-existing hygiene + guidelines fails).
- Four fidelity runs (§6 table) + one Phi story generate (fluent, 6.39 t/s).
- Temp corpus deleted; no branch, no edits; report committed; no push.

(End of file.)
