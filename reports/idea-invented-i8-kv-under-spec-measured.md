# MEASURED: i8 KV under speculation at 2k — slower than f16 (0.86-0.96x); the KV tier is not a spec multiplier

- **Idea**: Speculative decode at 2k spends 16-21% of the token in attention (per the 2k timer snapshot: matvec 74%, attn 21%, norm 5%). The i8 KV tier cuts attention traffic 4x at a priced 3.2e-4 KL, so pairing `--kv i8` with `--speculative` on a repetitive 2k prompt could multiply the spec win. Test f16-spec vs i8-spec vs i8-off on the reference 3B.
- **Cycle date**: 2026-11-11
- **Verdict**: **MEASURED — no, and i8 is slower under spec.** Llama-3.2-3B-Instruct-Q4_K_M, 2,050-token periodic prompt, `--context 2560 --n 32 --threads 10 --gpu off`:
  - f16 + spec: **30.63/32.41 t/s**; i8 + spec: **29.49/27.81 t/s** -> ratios **0.963/0.858** (median ~0.91). Same 6 steps and 5.17 accepted/step in every spec arm.
  - i8 + off: 13.32/10.46 t/s (vs f16-off ~11.5-12.5 in the same period) — i8 KV is neutral-to-noise on plain decode too.
  - Adding the priced 3.2e-4 KL of i8 KV buys nothing on this workload; f16 stays the right tier under speculation.
- **Tree state at close**: no code touched; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Llama-3.2-3B-Instruct-Q4_K_M.gguf |
| prompt | fox sentence x205 = 2,050 tokens |
| context / n / threads | 2560 / 32 / 10 |
| arms | `--kv f16 --speculative`, `--kv i8 --speculative`, `--kv i8` |
| pairs | 2, sampled consecutively in one window |

## 2. Results

| pair | f16 + spec | i8 + spec | ratio | i8 off |
|---:|---:|---:|---:|---:|
| 1 | 30.63 | 29.49 | 0.963 | 13.32 |
| 2 | 32.41 | 27.81 | 0.858 | 10.46 |
| **median** | 31.52 | 28.65 | **0.909** | 11.89 |

Spec line identical in all spec runs: 6 steps, 31 draft tokens accepted (5.17/step), 36 emitted.

## 3. Reading

- At 2k the i8 KV dequantisation cost outweighs the traffic saving on the spec path: 31 tokens in 6 verify steps means each step reads the KV cache only a handful of times relative to the weight stream, so halving KV bytes does not pay for per-block scale work. This mirrors the earlier conclusion that KV tiers are long-context levers, not short/medium-context ones.
- Speculation's 21% attention share at 2k is *not* dominated by KV bandwidth; it is more likely the per-head softmax and V accumulation work, which i8 does not remove.
- Usage: keep `--kv f16` (or f32) with `--speculative` at 2k; do not add i8 to chase a spec multiplier. Long-context (8k+) KV tier behavior was measured elsewhere and is out of this cycle's range.
- No ids were captured (plain output); the spec line's identical accept counts and the arms' identical token counts suggest no near-tie change, but this cycle does not claim id parity.

## 4. Reimplementation recipe

1. No code: `tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt "<fox x205>" --context 2560 --n 32 --threads 10 --gpu off --kv {f16,i8} [--speculative]`, alternating pairs.
2. Expect f16-spec 30-32 t/s, i8-spec 28-29 t/s, i8-off 10-13 t/s; i8-spec/f16-spec ~0.86-0.96.
3. Do not adopt i8 KV under speculation at this context; no follow-up spike is warranted without a long-context (8k+) profile showing KV bytes dominate.

## 5. Exact commands run and results

```text
tge generate ... --kv f16 --speculative  -> 30.63 / 32.41 t/s, 6 steps, 5.17/step
tge generate ... --kv i8  --speculative  -> 29.49 / 27.81 t/s, 6 steps, 5.17/step
tge generate ... --kv i8                 -> 13.32 / 10.46 t/s
git status --short                       -> clean (no code touched)
python aikit.py check aegean-tge         -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
