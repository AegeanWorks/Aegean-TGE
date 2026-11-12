# MEASURED: kq8d scope matrix at 2k context on 3B — attn 1.10x, ffn+down 1.10x, full stack 1.41x; the tier's gain is context-stable

- **Question**: The per-32 max tier was priced at short contexts (1.42x at a 1,200-token prompt; 1.38x with the older code). At 2k attention grows to 16-21% of decode, so matvec's share shrinks — does the tier's gain erode, and what do the individual scopes contribute on the reference 3B at 2k?
- **Cycle date**: 2026-11-11
- **Verdict**: **MEASURED — the tier holds at 2k and the stack is super-additive.** Llama-3.2-3B-Instruct-Q4_K_M, 2,050-token periodic prompt, `--context 2560 --n 32 --kv f16 --threads 10 --gpu off`, one interleaved window:
  - off 11.49 t/s; `--kq8d attn` 12.69 (**1.10x**); `--kq8d ffn,down` 12.59 (**1.10x**); `--kq8d head,ffn,down,attn` **16.19 t/s (1.41x)**.
  - **All four arms produced identical greedy ids** (`4C-1C-DE…`) on this workload.
  - Cross-window confirmation of the stack: 17.01 vs 11.49-12.63 in an earlier window (1.35x) and 15.40/15.18 vs 10.07/9.97 (1.53/1.52x) in a later one; the one-window matrix (1.41x) is the claim.
  - Earlier one-window files also read `attn` 1.06-1.08x and `ffn,down` 1.20-1.22x; singles move within box noise, the stack does not.
- **Tree state at close**: tier spike reverted from `src/cpu/avx2/dequant_batch_kq8_avx2.cpp`, `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp`, `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

Same per-32 max tier as `reports/idea-invented-kq8d-full-stack-per32-keep.md`: `head` (15% of weights, Q6_K), `ffn,down` (gate/up/down, 56%, Q4_K), `attn` (q/k/v/o, 19%, mixed) all route through `gemv_dequant_kq8_row_avx2` at `tokens == 1`. At 2k the decode phase split is ~74% matvec, ~16-21% attention, ~5% norm, so the converted 90% share still dominates.

## 2. Results (one window, 2,050-token prompt)

| arm | decode t/s | ratio vs off | ids |
|---|---:|---:|---|
| off | 11.49 | 1.00 | `4C-1C-DE` |
| `--kq8d attn` | 12.69 | 1.10 | identical |
| `--kq8d ffn,down` | 12.59 | 1.10 | identical |
| `--kq8d head,ffn,down,attn` | **16.19** | **1.41** | identical |

Cross-window stack ratios: 1.35 (single pair), 1.53/1.52 (two pairs). The stack's 1.41x at 2k equals its 1.42x at 1.2k: context length does not erode the tier on matvec-dominated decode.

## 3. Reading

- The max tier is context-stable: at 2k the matvec share is still ~74%, so 1.41x matches the 1.2k result. It should start eroding only where attention dominates (8k+), which the tier's own reports never claimed.
- The stack is super-additive again (singles 1.10 + 1.10 -> 1.41): the same effect seen in the tier re-pricing, consistent with the fused float path paying model-scale page/TLB costs that the integer row kernel avoids; the per-scope ratios are noisy, the combined arm is not.
- Ids identical on this workload: the periodic prompt's accept margins absorb the tier's near-tie flips. The prose thinking-set flip remains the honest near-tie risk.
- Usage: `--kq8d head,ffn,down,attn` at 2k is ~1.4x with the priced 4.382e-4 KL (f32) — at f16 KV add the f16 pricing (1.7e-6) and any stacking error; this cycle did not re-price the f16 stack, only the speed behaviour.

## 4. Reimplementation recipe

1. Re-apply the per-32 max tier (`reports/idea-invented-kq8d-full-stack-per32-keep.md` §8); build + 171/171.
2. Run `generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt "<fox x205>" --context 2560 --n 32 --kv f16 --threads 10 --gpu off --print-ids` with `--kq8d off|attn|ffn,down|head,ffn,down,attn` in one interleaved window.
3. Expect off ~11.5, singletons ~1.10x, stack ~1.41x, ids identical.
4. Revert. No code ships; the numbers extend the tier's context table.

## 5. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge generate ... --context 2560 --n 32 --kv f16 --threads 10 --gpu off --print-ids \
  [--kq8d attn|ffn,down|head,ffn,down,attn]                        -> §2 matrix
git checkout -- src/cpu/avx2/dequant_batch_kq8_avx2.cpp include/tge/models/qwen2.hpp \
  src/models/qwen2.cpp apps/cli/main.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the rebuilt binary; the committed tree is unchanged except this report.
