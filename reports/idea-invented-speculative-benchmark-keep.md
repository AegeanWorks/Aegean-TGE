# MEASURED: shipped speculative decoding on the standard benchmark — KEEP as a usage mode (1.95–2.15x on predictable prompts, ids identical)

- **Idea**: The engine has shipped a `--speculative` prompt-lookup drafter with SKIP-lock and fused verify since 2026-11-05, but every cycle in this loop benchmarked without it. Price it on the loop's standard repetitive prompt and on natural-language prompts.
- **Cycle date**: 2026-11-08
- **Verdict**: **KEEP as a usage mode (no code change).** On the periodic 4096-token prompt, decode is **3.79 → 7.38 t/s (1.95x)** and **3.89 → 8.38 t/s (2.15x)** with **5.09 draft tokens accepted per step** (ceiling 6) and **identical generated ids**. On natural-language prompts it is **0.98x and 0.92x** with identical ids — neutral-to-slightly-negative, as the historical log predicted. Enable it for predictable/repetitive continuations; keep it off by default for free-form text.
- **Tree state**: measurement-only; no code change, no branch, no spike. 145/145 tests and 4/4 gates on the current tree; nothing pushed.

## 1. Mechanism

`--speculative` enables prompt-lookup drafting: the drafter finds a suffix match of the recent history (40-token periodic window), proposes up to 5–6 tokens along the matched continuation, and the verifier accepts the longest matching prefix in one forward. SKIP-locked drafting and the fused steady state keep the bonus KV resident between full-accept steps. It is verifier-gated greedy, so accepted output is identical by construction. The CLI pins the per-head attention schedule for speculation (it prints a stderr warning) — the comparison below therefore includes a GQA → per-head change and still wins.

## 2. Falsifier

- **KEEP as a mode** if predictable prompts gain ≥1.5x with identical ids and natural prompts lose <10%.
- **Drop the recommendation** if natural prompts regress beyond that band.

Measured: 1.95–2.15x predictable; −2% and −8% natural.

## 3. Measured numbers

### Periodic 4096-token prompt, n=64, f32 KV, 6 threads

| pair | baseline | speculative | ratio | spec accepted | ids |
|---:|---:|---:|---:|---:|---|
| 1 | 3.79 t/s | **7.38 t/s** | **1.95x** | 5.09/step (56/11) | IDENTICAL |
| 2 | 3.89 t/s | **8.38 t/s** | **2.15x** | 5.09/step (56/11) | IDENTICAL |

The drafter emitted 66 tokens in 11 verify steps (5 accepted + 1 bonus per full-accept step). Acceptance is deterministic at 5.09/step on this prompt.

### Natural-language prompts, n=64, f32 KV

| prompt | baseline | speculative | ratio | ids |
|---|---:|---:|---:|---|
| "The quick brown fox … Explain why." | 6.64 t/s | 6.48 t/s | 0.98x | IDENTICAL |
| "Write a haiku about circuits dreaming." | 6.24 t/s | 5.76 t/s | 0.92x | IDENTICAL |

These match the historical log's story-prompt neutrality (0.97x); the per-head attention pin and drafter overhead cost a few percent when the suffix match does not lock.

## 4. Interpretation

- The loop's synthetic periodic benchmark is a best case for prompt-lookup speculation, and the gain is large: ~2x decode on top of everything else measured in this library. The standard benchmark numbers in earlier cycles (3.8–4.5 t/s at 4k) were baseline-mode numbers; the same workload with `--speculative` runs ~7.4–8.4 t/s.
- Natural text does not lock (0–2 accepted/step) and pays the scheduling/verify overhead, so the flag should be selected per workload, not defaulted.
- Speculation is orthogonal to the kq8d head/down tiers and the block prefill kernel, but this cycle did not combine them. Since speculation reduces the number of verifier forwards, kq8d's decode gain should shrink proportionally on locked prompts while the head's per-forward speedup still helps.

## 5. Disposition

**KEEP as a usage mode, report-only.** Recommend `--speculative` for code, templates, and repetitive or long-context workloads where the suffix history locks; leave it off for free-form generation. A future auto mode could enable it only when the drafter finds a suffix match (the drafter already computes that), avoiding the natural-text overhead.

## 6. Exact commands

```text
.\build\release\bin\tge_tests.exe                      → 145/145
python aikit.py check aegean-tge                        → 4/4
tge generate --model Llama-3B --tokens <4096 periodic ids> --n 64 \
  --threads 6 --gpu off --kv f32 --context 4352 [--speculative] --print-ids
tge generate --model Llama-3B --prompt "<natural>" --n 64 --threads 6 \
  --gpu off --kv f32 [--speculative] --print-ids
```

No push, no merge, no force-push. No code changed.
