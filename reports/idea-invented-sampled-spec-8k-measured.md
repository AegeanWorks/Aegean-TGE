# MEASURED: sampled speculation at 8k — 1.44-1.72x with the 6-row softmax now 53-55% of spec wall; the chat path holds

- **Idea**: Every 8k speculation number in the library is greedy (`idea-invented-spec-at-8k-measured.md`, `idea-invented-code-8k-spec-sparse-measured.md`). The common chat setting is sampled (`--temp 0.8 --top-p 0.9`), and sampled spec was only measured at 2k (2.28x, sampler ~25 ms/token; `idea-invented-sampled-spec-2k-measured.md`). At 8k the sampler runs the same six-row softmax per verify step while the forward phase grows; the question is whether the chat path keeps a real win and where its cost sits.
- **Cycle date**: 2026-11-13
- **Verdict**: **MEASURED — sampled speculation still pays at 8k, 1.44-1.72x, but the sampler is now the majority of the speculative time.** Llama-3.2-3B-Instruct-Q4_K_M, 8,193-token periodic prompt, `--context 8704 --n 32 --kv f16 --threads 10 --gpu off --temp 0.8 --top-p 0.9 --seed 7`, interleaved pairs:
  - **off**: **4.99 / 5.32 t/s**; decode `matvec 45%, attn 53-54%`; `timers sample 11.6-13.1 ms (0.2%)`.
  - **spec**: **8.58 / 7.64 t/s** -> **1.44-1.72x** (median ~1.58x); `7 steps, 23 accepted (3.29/step), 29 emitted`; `timers forward 3595-4040 ms, sample 1900-2248 ms` — the sampler is **53-55% of the speculative decode wall**, versus 0.2% in the off arm.
  - **Acceptance is slightly better than greedy on this prompt** (3.29 vs 3.00 accepted/step): the sampled accept rule consumes RNG differently, so per-step yield is not comparable token-for-token; distribution preservation is by construction and ids are not expected to match.
  - End-to-end the chat path keeps a real 8k win, but the growth path is now the sampler, not the forward: 7 verify steps softmax 6 rows each (~0.27-0.32 s per step), while the forward phase is 3.6-4.0 s for the whole run. Any future sampled-spec work at long context should attack the 6-row softmax/cut-selection cost, not the drafter.
- **Tree state at close**: no code touched (shipped flags); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Llama-3.2-3B-Instruct-Q4_K_M.gguf |
| prompt | `The fox runs. ` x 2048 = 8,193 tokens |
| context / n / KV / threads / GPU | 8704 / 32 / f16 / 10 / off |
| sampling | `--temp 0.8 --top-p 0.9 --seed 7` |
| arms | off, `--speculative` (interleaved: off->spec, spec->off) |

Sampled speculation does not reproduce greedy ids (the accept rule consumes randomness; it preserves the sampled distribution rather than the exact sequence), so this cycle makes no id-parity claim — timings and the `spec:`/`timers` lines only. Greedy parity was established separately (`idea-invented-code-8k-spec-sparse-measured.md`: sparse vs spec+sparse byte-identical).

## 2. Results

| pair | arm | decode t/s | ratio | spec line | timers (forward / sample) | phases (matvec / attn) |
|---|---|---:|---:|---|---|---|
| 1 | off | 4.99 | — | — | 6200.7 ms / 13.1 ms (0.2%) | 2808.2 ms (45%) / 3299.8 ms (53%) |
| 1 | spec | **8.58** | **1.72x** | 7 steps, 23 acc (3.29/step), 29 emitted | 3594.6 ms / 1899.9 ms (**52.6%**) | 1603.4 ms (45%) / 1898.0 ms (53%) |
| 2 | spec | **7.64** | **1.44x** | 7 steps, 23 acc (3.29/step), 29 emitted | 4039.5 ms / 2248.1 ms (**55.4%**) | 1828.4 ms (45%) / 2108.8 ms (52%) |
| 2 | off | 5.32 | — | — | 5815.4 ms / 11.6 ms (0.2%) | 2584.0 ms (45%) / 3142.8 ms (54%) |

Median of the four cross-pairings: **~1.58x**. The off arm is stable (4.99 / 5.32; attention phase 3.14-3.30 s), so the pair is admissible. The spec arm's matvec/attn totals are lower than the off arm's because 7 verify steps replace 31 single-token forwards — the batch amortizes weight passes — which is the same mechanism as the greedy path.

### Sampler cost, isolated

- Off: sample = 11.6-13.1 ms for 31 tokens (~0.4 ms/token).
- Spec: sample = 1,900-2,248 ms for 7 verify steps = **271-321 ms per step**, i.e. ~40-54 ms per verified row for the softmax + cut selection. This is ~53-55% of the speculative decode wall and ~2.7-3.5x the 2k sampler cost per step (the per-row work is context-independent; the extra is the six rows, not attention).
- Forward: spec 3.6-4.0 s vs off 5.8-6.2 s; the verify's six-row attention is inside this and still pays (attention 52-53% of the spec phase total, same share as off).

## 3. Reading

- **The chat path holds at 8k**: 1.44-1.72x is a real speedup on a sampled decode that would otherwise run at 5.0-5.3 t/s. It is lower than the 2k sampled ratio (2.28x) because the sampler share rises with verify width and the 8k baseline is slower, but it is not a loss.
- **The sampler is now the long-context sampled-spec bottleneck.** At 2k it was ~25 ms/token (~0.8 s for 32 tokens); at 8k the same per-step work is a larger fraction because there are fewer, wider verify steps and the forward is attention-heavy. A 6-row softmax is inherent to the verify width; the cut selection (top-p over 128k vocab, six times per step) is the candidate for future work.
- **Greedy vs sampled at 8k**: greedy spec is 1.42-1.61x on the same prompt (`idea-invented-spec-at-8k-measured.md`), sampled is 1.44-1.72x here — statistically the same band. The 8k verification cost dominates both, so mode choice does not change the long-context verdict.
- **Recommendation**: `--speculative` is valid for chat (`--temp 0.8 --top-p 0.9`) at 8k; quote **~1.5x**, not the 2k 2.28x. Pair with `--kv f16` (the speed+quality tier, `idea-invented-f16-vs-f32-8k-measured.md`). Adding `--sparse` on top is a separate, lossy-speed choice (13-19pp natural-text quality cost).
- **Sampler optimization is the named next lever** for sampled speculation at long context: the six-row top-p cut/softmax path is 53-55% of spec wall. It is not measured here; the existing sampler work (radix top-p, vectorized softmax, cascade) targeted single-row cuts, and a six-row batched cut would need a new spike.

## 4. Falsifier (stated before the runs)

**CONFIRM the chat path at 8k** if sampled spec is >= 1.20x off with a plausible accept rate. **QUALIFY/KILL** if it is within noise of off or slower. Measured 1.44-1.72x with 3.29 accepted/step: confirmed; the serving guidance is ~1.5x, not 2.3x.

## 5. File anchors (current HEAD, read-before-write verified)

- `apps/cli/main.cpp:1081-1195` — sampled speculative block: six-row `verify_draft_logits`, per-row penalized softmax, `speculative_accept`, bonus draw.
- `apps/cli/main.cpp:1185-1200` — the per-step correction forward that follows a partial accept.
- `src/models/qwen2.cpp` — `verify_draft_logits` (six rows x vocab) and the accept helper usage.
- `include/tge/models/qwen2.hpp` — `kSpeculativeBatch = 6`.
- `reports/idea-invented-sampled-spec-2k-measured.md` — the 2k sampled row (2.28x, sampler ~25 ms/token).
- `reports/idea-invented-spec-at-8k-measured.md` — the greedy 8k row (1.42-1.61x).
- `reports/idea-invented-f16-vs-f32-8k-measured.md` — the KV tier used for both arms.

## 6. Reimplementation recipe

1. Compact 8k prompt: `$p = ("The fox runs. " * 2048)`.
2. Interleave the arms (each ~4-5 min of prefill):
   ```powershell
   tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt $p \
     --context 8704 --n 32 --kv f16 --threads 10 --gpu off \
     --temp 0.8 --top-p 0.9 --seed 7 [--speculative]
   ```
3. Read `decode :`, `spec :`, `timers :`, `phases : decode`. Expect off 5.0-5.3 t/s (sample 0.2%); spec 7.6-8.6 t/s with 7 steps / 3.29 accepted per step and sampler 53-55% of the spec wall.
4. No id-parity check: sampled spec preserves the distribution, not the sequence. For a determinism check, run the same seed twice in one arm and compare ids within that arm.
5. No code ships.

## 7. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56, cycle gate)
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed

8k sampled (compact fox x2048, ctx 8704, n 32, f16, 10 threads, temp 0.8, top-p 0.9, seed 7):
  pair 1: off  4.99 t/s (sample 13.1 ms, 0.2%; attn 3299.8 ms, 53%)
          spec 8.58 t/s (1.72x; 7 steps, 23 acc, 3.29/step; sample 1899.9 ms, 52.6%)
  pair 2: spec 7.64 t/s (1.44x; same spec line; sample 2248.1 ms, 55.4%)
          off  5.32 t/s (sample 11.6 ms, 0.2%; attn 3142.8 ms, 54%)
git status --short                                -> clean (no code touched)
python aikit.py check aegean-tge                  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
