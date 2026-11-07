# INVENTED: kq8d head/combo — larger-corpus validation — KEEP, all three KV tiers qualify (f32 98.00%, f16 97.75%, i8 97.83%)

- **Idea**: Resolve the cycle-30 top-1 margins with a larger corpus. Cycle 30 measured 399 positions and found the combo exactly at the bar on f32 (390/399) and one/two flips short on f16 (389/399) and i8 (388/399). This cycle re-runs the combo at **1,199 positions** on a combined corpus and adds multi-prompt transcript parity.
- **Cycle date**: 2026-11-08
- **Verdict**: **KEEP — the combo qualifies on all three KV tiers.** At 1,199 positions: f32 **1,175/1,199 = 98.00%**, f16 **1,172/1,199 = 97.75%**, i8 **1,173/1,199 = 97.83%**, with mean KL **8.283e-4 / 8.034e-4 / 1.093e-3** — all under the 1.6e-3 bar and above the 97.6% top-1 precedent. The 399-position shortfalls were small-sample noise. Transcripts: **3/3 prompts × 64 generated tokens identical** to baseline.
- **Tree state**: spike reverted from `include/tge/models/qwen2.hpp` and `src/models/qwen2.cpp`; 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Mechanism

Re-applied the cycle-30 spike: `allow_kq8d` on `Qwen2::matvec`, `TGE_KQ8D_HEAD=1` for the 3072×128256 Q6_K LM head, `TGE_KQ8D_SCOPE=down` for the single-token FFN down projection. Prefill, attention and gate/up remain float.

The fidelity corpus is a concatenation of `docs/architecture.md`, `docs/measurement-log.md` and `README.md` (236 KB) in a temp file, tokenized by the engine. `--context 1536 --tokens 1200` yields **1,199 scored positions** — 3x the cycle-30 sample.

## 2. Falsifier

- **KEEP a tier** if mean KL ≤1.6e-3 and top-1 ≥97.6%-equivalent on the 1,199-position corpus, with transcript parity.
- **Restrict** if a tier stays below the top-1 precedent.

All three tiers pass; the previous f32-only restriction is lifted.

## 3. Measured numbers

### Fidelity, 1,199 positions (context 1536)

| KV | mean KL | worst KL | top-1 | wall |
|---|---:|---:|---:|---:|
| f32 combo | **8.283e-4** | 1.682e-2 | **1,175/1,199 (98.00%)** | 631 s |
| f16 combo | **8.034e-4** | 1.031e-2 | **1,172/1,199 (97.75%)** | 681 s |
| i8 combo | **1.093e-3** | 9.680e-3 | **1,173/1,199 (97.83%)** | 773 s |

### Cycle-30 (399 positions) vs this cycle (1,199 positions)

| tier | 399-pos top-1 | 1,199-pos top-1 | movement |
|---|---:|---:|---:|
| f32 | 390/399 (97.74%) | 1,175/1,199 (98.00%) | +0.26 pt |
| f16 | 389/399 (97.49%) | 1,172/1,199 (97.75%) | +0.26 pt |
| i8 | 388/399 (97.24%) | 1,173/1,199 (97.83%) | +0.59 pt |

The three tiers move in the same direction; the 399-position differences were noise from a short corpus, not a real tier ordering (i8 now has the second-best top-1 despite the highest KL).

### Transcript parity (f32 KV, `--n 64 --print-ids`)

| prompt | tokens matched | first divergence |
|---|---:|---:|
| "The quick brown fox jumps over the lazy dog. Explain why." | 64/64 | none |
| "Write a haiku about circuits dreaming." | 64/64 | none |
| "List three reasons a compiler might optimize a loop, then summarize." | 64/64 | none |

## 4. Interpretation

- The combo's fidelity cost is stable at this sample size: mean KL 8.0e-4–1.1e-3, top-1 97.75–98.00%.
- The f16/i8 margins are still thin (~2–3 flips above the 97.6% precedent), unlike f32 (~5 flips). Keep f32 as the primary/manual recommendation; f16 and i8 qualify on this corpus but should be re-checked on any larger evaluation.
- i8's higher KL with slightly better top-1 than f16 is consistent with KV quantization noise occasionally breaking near-ties in the reference's favor; top-1 is not monotone in mean KL at these scales.
- Transcripts agree exactly on three short prompts; this is necessary but not sufficient evidence for a default-on decision (64 tokens × 3 prompts is small).

## 5. Reimplementation recipe

Unchanged from cycle 30:

1. `allow_kq8d` on `matvec`; kq8d branch before `gemv_dequant_batch` for Q4_K/Q6_K with AVX2 and `!scalar_kernels_`.
2. Head call site in `output_logits`; down call site in single-token `forward`.
3. Leave prefill, attention and gate/up float.
4. Gate: the 1,199-position corpus above, transcript parity, 145/145, 4/4.

## 6. Disposition

**KEEP, report-only.** The combo is validated on a 3x larger corpus with all three KV tiers above the bar and exact transcripts. The tier recommendation is revised from cycle 30: **f32 primary, f16/i8 qualified but with ~2-flip margins**. Integration remains the human-gated decision.

## 7. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
# corpus: temp concat of docs/architecture.md + docs/measurement-log.md + README.md
TGE_KQ8D_HEAD=1 TGE_KQ8D_SCOPE=down tge fidelity --model Llama-3B \
  --corpus <big corpus> --context 1536 --tokens 1200 --threads 6 --gpu off \
  --kv f32|f16|i8
TGE_KQ8D_HEAD=1 TGE_KQ8D_SCOPE=down tge generate --model Llama-3B \
  --prompt "<three prompts>" --n 64 --threads 6 --gpu off --print-ids
git checkout -- include/tge/models/qwen2.hpp src/models/qwen2.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
