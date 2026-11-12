# MEASURED: speculation at 4k context on 3B — ~1.9-2.1x (down from 2.74x at 2k); attention share reaches 29-33%

- **Question**: The spec context table had 2,050 tokens (2.74x) and 6,600-token decode/tier data, but no 4k spec point. At 4k the verify's 6-row attention reads a 4k KV cache, so the step cost should rise and the spec ratio compress. Measure it on the reference 3B.
- **Cycle date**: 2026-11-11
- **Verdict**: **MEASURED — speculation still pays at 4k, at ~2x.** Llama-3.2-3B-Instruct-Q4_K_M, 4,100-token periodic prompt, `--context 4608 --n 16 --kv f16 --threads 10 --gpu off`:
  - off: 10.27/11.32 t/s; phases `matvec 69-70%, attn 28-29%, norm 2%`.
  - spec: **23.65/20.71/21.24 t/s** across three runs; `3 steps, 16 draft tokens accepted (5.33/step), 18 emitted`; phases `matvec 62%, attn 33%, norm 4%`.
  - ratios: 2.09 (23.65/11.32), 2.02 (20.71/10.27), 1.88 (21.24/11.32) -> **median ~2.02x**.
  - vs the 2k point (2.74x): the drop tracks attention's share (21% -> 33% of decode) as the verify reads 2x the KV per step.
- **Tree state at close**: no code touched; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Llama-3.2-3B-Instruct-Q4_K_M.gguf |
| prompt | fox sentence x410 = 4,100 tokens |
| context / n / threads / KV | 4608 / 16 / 10 / f16 |
| arms | off vs `--speculative` |
| runs | off 2, spec 3 (spec came from three separate launches; same seed/prompt) |

Note on capture: `--prompt` with a 4.1k-token text plus `2>&1 | Out-File` truncated the spec arm in the loop harness (a PowerShell stream quirk); the numbers above were captured with `Start-Process ... -RedirectStandardOutput` and by a direct run, all with the same arguments.

## 2. Results

| run | off t/s | spec t/s | ratio | spec line | decode phases |
|---|---:|---:|---:|---|---|
| a | 11.32 | 23.65 | 2.09 | 3 steps, 5.33/step, 18 emitted | matvec 62%, attn 33% |
| b | 10.27 | 20.71 | 2.02 | 3 steps, 5.33/step, 18 emitted | - |
| c | 11.32 | 21.24 | 1.88 | 3 steps, 5.33/step, 18 emitted | - |
| **median** | 11.32 | 21.24 | **2.02** | | |

Off phases: `matvec 1006.8/928.6 ms (69/70%), attn 419.3/365.4 ms (29/28%), norm 29.1/26.4 ms (2%)`.

## 3. Reading

- Spec's context curve on 3B: **2.74x at 2k, ~2.0x at 4k**. The step count halves (16 tokens in 3 steps) but each verify step costs more because attention is now a third of decode; the ratio compresses by roughly the attention-share growth.
- The trend implies ~1.5-1.6x at 8k if attention keeps growing linearly — still worth it on repetitive prompts, but the gap to plain decode narrows. The 6.6k tier table (1.21x for the kq8d tier) and this spec table should be read together: at long context both levers decay, and the long-context levers are the KV tiers/sparse attention instead.
- Accept rate is unchanged (5.33/step vs 5.17 at 2k), so the compression is verify cost, not draft quality. Usage: keep `--speculative` on repetitive prompts at 4k; expect ~2x, not ~3x.
- No fidelity run owed: greedy spec is verifier-exact on this workload; the `spec` line's identical accept counts and 18-token output match the 2k shape (proportionally).

## 4. Reimplementation recipe

1. No code: `tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt "<fox x410>" --context 4608 --n 16 --kv f16 --threads 10 --gpu off [--speculative]`.
2. Expect off ~10-11.5 t/s (attn 28-29%), spec ~20.7-23.7 t/s with 3 steps and 5.33 accepted/step; ratio ~1.9-2.1x.
3. Compare with the 2k result (`idea-invented-spec-at-2k-measured.md`, 2.74x) to read the attention-share compression.
4. No code ships.

## 5. Exact commands run and results

```text
tge generate ... --prompt "<fox x410>" --context 4608 --n 16 --kv f16 --threads 10 --gpu off
  -> 10.27 / 11.32 t/s, matvec 69-70%, attn 28-29%
tge generate ... --speculative
  -> 23.65 / 20.71 / 21.24 t/s, 3 steps, 16 accepted (5.33/step), 18 emitted
git status --short                 -> clean (no code touched)
python aikit.py check aegean-tge   -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
