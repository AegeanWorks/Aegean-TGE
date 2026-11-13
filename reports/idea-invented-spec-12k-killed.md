# MEASURED + KILLED: speculation at 12k is a loss on the only fit-capable 12k prompt (0.84x) — acceptance collapses to 1.50/step while attention is 65-67%

- **Idea**: The spec context curve is measured at 2k (2.74x), 4k (2.02x), and 8k (1.42-1.61x periodic, 1.18-1.25x prose, 1.13x code). There is no point between 8k and the model limit, and at 12k the verify attention share rises to ~65% of decode. Does the 8k win keep compressing, hold, or flip? Build the largest periodic prompt that fits the Windows command line (`a` + ` a`x12,288 = 12,289 tokens) and A/B off vs `--speculative`.
- **Cycle date**: 2026-11-13
- **Verdict**: **KILLED — speculation loses at 12k on this workload: 3.49 vs 4.15 t/s (0.84x).** Llama-3.2-3B-Instruct-Q4_K_M, 12,289-token period-1 lock prompt, `--context 12800 --n 16 --kv f16 --threads 10 --gpu off`, prefill matched to 0.2%:
  - **off**: prefill **407.0 s** (matvec 206.7, attn 182.2), decode **4.15 t/s** (attn 2417 ms, **67%**).
  - **spec**: prefill **407.7 s** (matvec 205.0, attn 184.8), decode **3.49 t/s** (0.84x); `4 steps, 6 draft tokens accepted (1.50/step), 10 emitted`; decode attn 2767 ms (64%).
  - A second spec run in a slower window reproduced the spec line exactly (1.50/step) at **3.33 t/s** — the acceptance collapse is deterministic on this prompt, not noise. (The matching off arm of that pair was killed by the 15-minute shell cap; the first pair, with prefill matched to 0.2%, is the quoted ratio.)
  - **The mechanism is acceptance, not verify cost.** The six-row verify shares the KV read: per-row decode attention is 692/6 = 115 ms versus 161 ms per single-row token, the same amortization as 8k. But 1.50 accepted/step cannot pay for four verify steps when the off arm does 15 single-token forwards. The 8k version of the same prompt accepted **4.33/step**; the model's period-1 continuation decays with context length, and the verifier correctly rejects what the drafter proposes.
  - **Spec's context boundary is workload-limited, not context-limited**: at 8k, periodic/fox accepts 3.00-5.33/step (1.4-1.6x), code 3.00 (1.13x), prose 3.17 (1.18-1.25x); at 12k the only measurable periodic prompt accepts 1.50 and loses. Do not extrapolate an 8k spec win to 12k without measuring accepted/step at that context.
- **Tree state at close**: no code touched (shipped flags); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Llama-3.2-3B-Instruct-Q4_K_M.gguf |
| prompt | `a` + ` a` repeated 12,288 times = **12,289 tokens**, 24,577 chars (the densest periodic prompt that fits the ~32.7k command-line limit) |
| context / n / KV / threads / GPU | 12800 / 16 / f16 / 10 / off |
| arms | off, `--speculative` |
| runs | off 1, spec 2 (second in a drifted window) |

At 12k the f16 KV cache is ~1.47 GB; the run loaded the full 12,800 context, so this is not a capacity-limited measurement.

## 2. Results

| run | arm | prefill (matvec / attn) | decode t/s | decode attn | spec line |
|---|---|---|---:|---|---|
| 1 | off | 407.0 s (206.7 / 182.2) | **4.15** | 2417 ms (67%) | — |
| 2 | spec | 407.7 s (205.0 / 184.8) | **3.49 (0.84x)** | 2767 ms (64%) | 4 steps, 6 acc (1.50/step), 10 emitted |
| 3 | spec | 449.1 s (— / —) | 3.33 | 2926 ms (65%) | 4 steps, 6 acc (1.50/step), 10 emitted |

Run 3's prefill was 10% slower (box drift) and its off partner was cut off, so run 3 is confirmation of the spec line only. Run 1 vs run 2 is the admissible pair: prefill totals differ by 0.2% and matvec by 0.8%.

### Where the 12k time goes

- Prefill: **407 s**, matvec 206.7 s (51%), attention 182.2 s (45%), norm 15.1 s (4%). Attention was 66-69 s of ~210 s at 8k (33%); the quadratic term is now visible and prefill throughput is **30.1 t/s** (8k: 38-40 t/s).
- Decode: attention is **67%** of the phase total at 12k (8k: 54-56%), matvec 32%.
- Spec: four verify steps cost 4.30 s and emit 10 tokens before the chain exits; the off arm emits 15 tokens in 3.61 s.

## 3. Reading

- **The 8k spec recommendation is bounded at 8k-style workloads.** Speculation is a quality-neutral accelerator when the drafter accepts ~3+ tokens/step; at 12k on this prompt it accepts 1.50 and the verify's six rows cannot be amortized. The 12k decode is 65-67% attention, so even a 3-accepted step would only roughly break even; the five-draft chain needs acceptance close to its 8k values to win.
- **Acceptance decay is model/data behaviour, not an engine regression.** The drafter proposes the period-1 continuation; the verifier (exact greedy) rejects it because the model stops repeating " a" at 12k. No gate or policy fixes a correct rejection; the lever is the drafter quality, which is a different project.
- **Prefill is now the dominant 12k cost**: 407 s at 30 t/s, 45% of it attention, versus 205-240 s at 8k. The only measured prefill levers at long context are the lossy sparse path (per-token: 1.06-1.20x; block-sparse recipe: ~1.27x) — consistent with the 8k decision matrix, and none is quality-preserving.
- **Guidance**: at 12k, default to f16 + dense without speculation unless the workload is shown to accept drafts at that context; use `--speculative` only after checking the `spec:` accepted/step on a sample. For 8k and below, keep the measured recommendations.
- **The context curve is therefore not monotone in a simple way**: the ratio falls with context at fixed acceptance (2k 2.74x -> 8k 1.5x) and can fall below 1.0 when acceptance itself collapses (12k 0.84x). Quote both axes: context and accepted/step.

## 4. Falsifier (stated before the runs)

**EXTEND spec's viability to 12k** if spec is >= 1.10x off on the 12k prompt with a plausible accept rate. **KILL** if it is <= 1.0x or acceptance is structurally low. Measured 0.84x at 1.50 accepted/step: killed. The result is specific to the measured prompt; a 12k workload with 3+ accepted/step would need its own run.

## 5. File anchors (current HEAD, read-before-write verified)

- `apps/cli/main.cpp:788-922` — fused greedy chain measured (six-row verify, `position = vpos + 7` on full accept).
- `apps/cli/main.cpp:924-1070` — non-fused path (not used; default is fused).
- `include/tge/models/qwen2.hpp:180` — `kSpeculativeBatch = 6`.
- `src/cpu/avx2/attention_avx2.cpp:660-864` — the block kernel whose six-row KV sharing sets the verify cost.
- `reports/idea-invented-spec-at-8k-measured.md` — the 8k periodic curve point.
- `reports/idea-invented-spec-prose-8k-kills-lookup-gate.md`, `idea-invented-code-8k-spec-sparse-measured.md` — the 8k prose/code points.
- `reports/idea-invented-8k-decision-matrix-closeout.md` — the matrix this bounds in context.

## 6. Reimplementation recipe

1. Build the prompt (fits the command line):
   ```powershell
   $p = ("a" + (" a" * 12288))   # 12,289 tokens, 24,577 chars
   ```
   Confirm with `--context 64 --n 1`: `prompt : 12289 tokens`.
2. Run the arms (each ~7-8 min of prefill; the off+spec pair must be within one shell call or split at ~7.5 min each to avoid the 15-minute cap):
   ```powershell
   tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt $p \
     --context 12800 --n 16 --kv f16 --threads 10 --gpu off [--speculative]
   ```
   Expect off 4.15 t/s (prefill 407 s, attn 67%); spec 3.49 t/s (4 steps, 1.50 accepted/step, 10 emitted). If the box drifts, quote only pairs with prefill within ~3%.
3. Check the `spec:` accepted/step before trusting any 12k spec configuration.
4. No code ships.

## 7. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56, cycle gate)
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed

12k lock prompt (ctx 12800, n 16, f16, 10 threads, gpu off):
  off  : prefill 407.0 s (matvec 206.7, attn 182.2) ; decode 4.15 t/s (attn 2417 ms, 67%)
  spec : prefill 407.7 s (matvec 205.0, attn 184.8) ; decode 3.49 t/s (0.84x ; 4 steps, 6 acc, 1.50/step, 10 emitted)
  spec (drifted window) : 3.33 t/s, same 1.50/step line ; matching off arm cut off by the 15-min shell cap
git status --short                                -> clean (no code touched)
python aikit.py check aegean-tge                  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
