# MEASURED: kq8d max tier context table — 1.41x at 2k, 1.33x at 4k, 1.21x at 6.6k; erosion is real but the tier stays worth it to 8k

- **Question**: `idea-invented-kq8d-scope-matrix-2k-measured.md` showed the max tier is 1.41x at 2k, matching its 1.42x at 1.2k, and predicted erosion only once attention dominates. Where does that start, and how far does it go on the reference 3B?
- **Cycle date**: 2026-11-11
- **Verdict**: **MEASURED — a clean erosion curve, no cliff.** Llama-3.2-3B-Instruct-Q4_K_M, `--kv f16 --threads 10 --gpu off`, greedy, `--kq8d head,ffn,down,attn` vs off:
  - **2,050 tokens**: 11.49 -> 16.19 t/s (**1.41x**) — the prior matrix.
  - **4,100 tokens** (`--context 4608`, n=16, two pairs): 9.28/9.65 -> 12.33/12.77 t/s (**1.33/1.32x**).
  - **6,600 tokens** (`--context 8704`, n=16, one pair): 6.32 -> 7.64 t/s (**1.21x**).
  - Baseline decode itself falls 11.49 -> 9.28 -> 6.32 t/s as attention grows; the tier's gain tracks the shrinking matvec share (~1.3-1.4%/1k tokens of context).
- **Tree state at close**: tier spike reverted from `src/cpu/avx2/dequant_batch_kq8_avx2.cpp`, `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp`, `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

Same per-32 max tier: head (Q6_K, 15% of weights) + gate/up/down (Q4_K, 56%) + q/k/v/o (mixed, 19%) through `gemv_dequant_kq8_row_avx2` at `tokens == 1`. Decode time = matvec (context-independent) + attention (linear in context) + norm. The tier converts the matvec share; as context grows, attention's share grows and the ratio decays toward 1.

## 2. Results

| context tokens | off t/s | tier t/s | ratio | notes |
|---:|---:|---:|---:|---|
| 2,050 | 11.49 | 16.19 | 1.41 | one window, matrix cycle |
| 4,100 | 9.28/9.65 | 12.33/12.77 | 1.33/1.32 | two pairs, 81-91 s prefill each |
| 6,600 | 6.32 | 7.64 | 1.21 | one pair, 160-172 s prefill each |

Prompt shapes differ by context point (fox x410 at 2-4k, "one..ten" x600 at 6.6k); the variable is context length, not prompt content, and only same-pair ratios are claimed. Prefill times are recorded (48.3/46.1 vs 44.8/45.3 tok/s at 4k; 41.2 vs 38.5 at 6.6k) — the tier is decode-only; prefill differences are noise.

## 3. Reading

- The tier remains a net win across the whole tested range: 1.41x at 2k, 1.33x at 4k, 1.21x at 6.6k. Even at 6.6k the converted matvec share is worth more than the attention share it cannot touch.
- The erosion slope (~0.03x per 1k tokens in this range) predicts ~1.1x at 10k and parity around 18k if attention kept growing linearly; the tier's contracts never promised the long-context regime, and the shipped KV tiers (f16/i8, sparse attention) are the long-context levers instead.
- Practical guidance for the README/tiers: `--kq8d head,ffn,down,attn` is ~1.4x at 2k, ~1.3x at 4k, ~1.2x at 6-7k on 3B with the priced 4.382e-4 KL (f32); it stays worth enabling on matvec-bound prompts at any context up to at least 7k.
- Ids were not re-checked at 4k/6.6k in this cycle (the matrix cycle verified them at 2k); a 16-token greedy run at 4k could be added to the promotion checklist if the tier ships.

## 4. Reimplementation recipe

1. Re-apply the per-32 max tier (`reports/idea-invented-kq8d-full-stack-per32-keep.md` §8); build + 171/171.
2. 4k: `generate --prompt "<fox x410>" --context 4608 --n 16 --kv f16 --threads 10 --gpu off [--kq8d head,ffn,down,attn]`, two alternating pairs -> expect 1.32-1.33x.
3. 6.6k: `--prompt "<one..ten x600>" --context 8704 --n 16`, one pair -> expect ~1.21x.
4. Revert; the numbers extend the tier's context table.

## 5. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge generate ... --prompt "<fox x410>" --context 4608 --n 16 --kv f16 --threads 10 \
  --gpu off [--kq8d head,ffn,down,attn]                            -> 1.33/1.32x
tge generate ... --prompt "<one..ten x600>" --context 8704 --n 16 --kv f16 --threads 10 \
  --gpu off [--kq8d head,ffn,down,attn]                            -> 1.21x
git checkout -- src/cpu/avx2/dequant_batch_kq8_avx2.cpp include/tge/models/qwen2.hpp src/models/qwen2.cpp apps/cli/main.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the rebuilt binary; the committed tree is unchanged except this report.
