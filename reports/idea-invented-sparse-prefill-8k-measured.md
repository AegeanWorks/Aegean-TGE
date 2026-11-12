# INVENTED: strided sparse prefill at 8k — MEASURED (attention 0.61-0.83x, total prefill 1.06-1.15x; per-token fallback forfeits block reuse, so prefill gains ~1/3 of decode's)

- **Idea**: The decode reopening (`reports/idea-invented-sparse-8k-reopened.md`) priced `--sparse` at 8k on the reference 3B and found 1.36-1.77x plain decode. The prefill side was never priced at 8k on the shipped f16 strided kernel: the recorded prefill evidence stops at 2k (f32, 0.49-0.52x attention on Llama-3B) and 4k (f16 0.55-0.70x attention on 0.5B). At 8k, dense prefill spends 66-69 s of ~205-213 s in attention, and sparse prefill uses the per-token fallback (`attention_batch`'s block path is gated off when sparse is enabled). Measure whether the decode win survives in prefill, and where the per-token fallback costs.
- **Cycle date**: 2026-11-12
- **Verdict**: **MEASURED — sparse prefill wins at 8k, but the per-token fallback costs about half the theoretical saving.** Llama-3.2-3B-Instruct-Q4_K_M, 8,193-token prompt (`The fox runs. ` x 2048), `--context 8704 --n 16 --kv f16 --threads 10 --gpu off`, three interleaved pairs:
  - **Total prefill**: dense **207.4 / 213.0 / 205.1 s** vs sparse **195.0 / 196.7 / 178.0 s** -> ratios **1.063 / 1.083 / 1.152** (median 1.083). The cleanest pair has matched matvec phases (127.2 vs 126.1 s) and reads **1.152x**.
  - **Prefill attention**: dense **68.8 / 69.3 / 66.7 s** vs sparse **44.6 / 57.2 / 40.6 s** -> ratios **0.648 / 0.826 / 0.609** (median 0.648).
  - The kept fraction at 8,193 tokens is ~30% (`4 + (8193-516)/4 + 512 = 2,435` of 8,193), but the measured attention ratio is 0.61-0.83, i.e. per-token sparse is ~2-2.8x less efficient per kept row than the dense 128-query block kernel. The gap is the block kernel's 128-query reuse, which the per-token fallback cannot have.
  - **Whole-run reading**: at 8k prefill dominates wall time (205 s vs 1.6-2.8 s for 16 decode tokens). The combined sparse run is ~1.16x end-to-end at short outputs; the 1.36-1.77x decode ratio applies only to long generations. Prefill is the 8k lever for short outputs.
- **Tree state at close**: no code touched (both flags shipped); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Mechanism (current tree, read before measuring)

- With `--sparse`, `Qwen2::attention_batch` skips the 128-query block kernel (`src/models/qwen2.cpp:1584`, gate `!sparse_.enabled`) and falls through to `pool.parallel_for(tokens * heads)` calling `attention_head` per (token, head) (`:1635-1655`). `attention_head` then dispatches the strided sparse kernel (`:1331-1341`).
- The dense block kernel loads each K/V row once and dots it with up to 128 query rows. The per-token sparse path loads each kept K/V row once per token. At 8k the kept set is ~30% of the span, so the sparse path reads `0.30 x span` rows per token against the block path's `span` rows per 128 tokens; the kept-fraction saving is real, but the lost reuse means sparse reads `~0.30 x 128 = 38` kept-row-equivalents per token versus one dense row per token per 128-token chunk — still less traffic, but with per-call overhead and no batched softmax/store.
- Dequant/scale is not the issue here: the f16 strided kernel converts on load, and the attention phase numbers scale with the kept count, not with the query count.
- The shipped prefill guidance ("kept opt-in for prefill-bound long prompts") was based on 2k/4k evidence where the sparse per-token path still beat dense; this cycle prices the 8k cell.

## 2. Results

All runs: 3B Q4_K_M, `--kv f16 --threads 10 --gpu off`, prompt 8,193 tokens, n=16. Matvec should be context- and sparse-independent; it is shown to expose box drift. Pairs alternate order (D->S, S->D, D->S).

| pair | dense total / attn | sparse total / attn | total ratio | attn ratio | matvec dense / sparse |
|---|---|---:|---:|---:|---|
| 1 (D->S) | 207.4 s / 68.8 s | 195.0 s / 44.6 s | **1.063** | 0.648 | 127.2 / 138.1 s |
| 2 (S->D) | 213.0 s / 69.3 s | 196.7 s / 57.2 s | **1.083** | 0.826 | 131.9 / 128.0 s |
| 3 (D->S) | 205.1 s / 66.7 s | 178.0 s / 40.6 s | **1.152** | 0.609 | 127.2 / 126.1 s |
| **median** | **207.4 / 68.8** | **195.0 / 44.6** | **1.083** | **0.648** | |

Pair 3 is the cleanest pricing: the matvec phases agree to 0.9%, so the 27.1 s total drop is attributable to the 26.1 s attention drop (ratio 0.609). Pair 1's sparse matvec ran 8.6% high (box drift), which is why its total ratio understates the effect. Pair 2's sparse attention (57.2 s) is the noisy outlier in an otherwise consistent sparse arm (44.6, 40.6 s).

## 3. Reading

- **Sparse prefill pays at 8k**: attention is 0.61-0.83x dense, total prefill 1.06-1.15x. The gain is real but roughly a third of decode's (1.36-1.77x), and the reason is structural: `--sparse` replaces the block kernel with per-token calls, so the kept-fraction win is paid for with lost 128-query reuse.
- **Block-sparse prefill is the unfilled lever**: keep the 128-query block kernel and apply the sink/stride/window kept set inside it (block-level kept set, one K/V row load feeding the whole query block). Predicted attention from the measured per-row cost: the per-token path is ~2-2.8x less efficient than block dense, so a block-sparse kernel at 30% kept should land near **0.30-0.35x** dense attention instead of 0.61-0.83x, i.e. another ~2x on the prefill attention phase and a predicted 1.30-1.40x total prefill. No block-sparse kernel or report exists in the tree (checked by name and measurement-log grep); this is a new recipe, not a retry.
- **Usage guidance**: at 8k with short outputs, the prefill dominates (~205 s vs 1.6-2.8 s for 16 tokens), so the total-run win from `--sparse` is ~1.16x even though the decode phase is 1.36-1.77x faster. For long generations the decode ratio takes over. The opt-in prefill guidance extends to 8k; the previously quoted +13% at 4k is still the best prefill case.
- **Speculation composes with sparse in prefill too**, because the verifier is gated onto the per-token sparse path; that is the decode-side stacking measured last cycle and it is not re-priced here.
- No fidelity run owed: no arithmetic change; sparse is the already-priced lossy tier (KL 2.6e-2 class) and both arms are the shipped opt-in path.

## 4. Falsifier (stated before the runs)

**EXTEND the prefill gate to 8k** if sparse prefill is >= 1.10x dense with a matched matvec phase or a consistent direction across interleaved pairs. **KEEP the gate at <= 4k** otherwise. Measured 1.063 / 1.083 / 1.152 with attention 0.61-0.83x in every pair: the direction is unanimous and the matched-matvec pair clears the bar; the prefill gate extends, with the per-token caveat recorded.

## 5. File anchors (current HEAD, read-before-write verified)

- `src/models/qwen2.cpp:1584` — `attention_batch` dense block gate (`!sparse_.enabled`), the reason sparse prefill is per-token.
- `src/models/qwen2.cpp:1635-1655` — the `tokens * heads` per-token fallback that sparse/ i8 prefill take.
- `src/models/qwen2.cpp:1329-1342` — sparse dispatch in `attention_head` (strided f32/f16).
- `src/cpu/avx2/attention_avx2.cpp:263,367` — the two strided kernels.
- `reports/idea-invented-sparse-8k-reopened.md` — the decode-side 8k reopening this cycle complements.
- `reports/idea-invented-sparse-nogather-llama-measured.md`, `...-sparse-f16-proceed.md` — the 2k prefill evidence (0.49-0.52x f32, 0.55-0.70x f16) this cycle extends to 8k.

## 6. Reimplementation recipe

1. No code: shipped flags.
2. Compact 8k prompt: `$p = ("The fox runs. " * 2048)` (8,193 tokens).
3. Interleave the arms; each run costs ~3-4 min of prefill on this box:

   ```powershell
   tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt $p --context 8704 --n 16 --kv f16 --threads 10 --gpu off [--sparse]
   ```

4. Read the `prefill :` and `phases : prefill` lines. Expect dense 205-213 s total (attn 66-69 s, matvec 127-132 s) and sparse 178-197 s (attn 41-57 s, matvec 126-138 s); ratio 1.06-1.15, attention ratio 0.61-0.83. The matched-matvec pair (127.2 vs 126.1 s) is the clean pricing.
5. For the block-sparse recipe (not implemented): in `attention_batch`, keep the block kernel when `sparse_.enabled` and pass the sink/stride/window kept set into it; the kernel's per-key loop must walk the three segments instead of `[0, start_position + token)` while keeping the 128-query dot tile. Predicted 0.30-0.35x attention, 1.30-1.40x total prefill; the differential test is against the per-token sparse path with identical kept sets.
6. No code ships this cycle.

## 7. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56, from the cycle gate)
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed

8k prefill pairs (compact fox x2048, ctx 8704, n 16, f16, 10 threads):
  pair 1  D->S : dense 207.4 s (attn 68.8, matvec 127.2) ; sparse 195.0 s (attn 44.6, matvec 138.1)
  pair 2  S->D : sparse 196.7 s (attn 57.2, matvec 128.0) ; dense 213.0 s (attn 69.3, matvec 131.9)
  pair 3  D->S : dense 205.1 s (attn 66.7, matvec 127.2) ; sparse 178.0 s (attn 40.6, matvec 126.1)
git status --short                                -> clean (no code touched)
python aikit.py check aegean-tge                  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
