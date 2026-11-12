# INVENTED: strided sparse decode at 8k — REOPENED (1.36-1.77x plain, attention 3.0-3.8x faster; spec+sparse stacks to 1.6-1.75x over spec, ~2.2-3.0x over dense)

- **Idea**: `reports/idea-invented-sparse-8k-killed.md` killed sparse decode at 8k (−13 to −23%) because the per-token `gather` copied the kept rows and the copy cost scaled with context. That report explicitly named the no-gather strided port as the reopen trigger. The port has since shipped (`winner-port-sparse-nogather-keep.md`, `winner-port-sparse-f16-keep.md`) and was re-validated at ~2k history, but the shipped strided kernel has never been priced at 8k — where decode attention is now 52-56% of the time (`idea-invented-spec-at-8k-measured.md`). Re-run the 8k A/B on the reference 3B with f16 KV.
- **Cycle date**: 2026-11-12
- **Verdict**: **MEASURED — the 8k kill is reversed by the strided kernel; the long-context sparse gate reopens.** Llama-3.2-3B-Instruct-Q4_K_M, 8,193-token periodic prompt (`The fox runs. ` x 2048), `--context 8704 --n 16 --kv f16 --threads 10 --gpu off`, interleaved pairs:
  - **Plain decode**: dense **5.40 / 7.04 t/s** (attn 54%, 1150-1508 ms) vs sparse **9.52 / 9.57 t/s** (attn 24-25%, 377-400 ms) -> **1.36-1.77x**, sparse arm stable.
  - **Speculation composes**: spec **9.68 / 9.37 t/s** (4 steps, 3.00 accepted/step, attn 52-53%) vs spec+sparse **15.61 / 16.36 t/s** (3 steps, 5.33 accepted/step, attn 34-35%) -> **1.61-1.75x over spec**; against dense plain decode that is **~2.2-3.0x**.
  - The 2k control on the same model shows the shipped near-2k gate still holds: dense 11.39/11.78 vs sparse 12.19/12.43 -> **1.05-1.07x**, attention 16% -> 10%.
  - Sparse is the priced lossy tier, not id-preserving: on the 2k periodic control the ids diverge at position 10 (dense `... 13 578`, sparse `... 13 220`) and re-converge to the dense sequence afterwards.
- **Tree state at close**: no code touched (both flags shipped); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Mechanism (current tree, read before measuring)

- The three shipped sparse pieces are in the tree: `attention_head_static_sparse_avx2` (`src/cpu/avx2/attention_avx2.cpp:263`) and its f16 twin (`:367`), dispatched in `Qwen2::attention_head` (`src/models/qwen2.cpp:1331-1341`) when `sparse_.enabled && !kv_q8_ && span > sink + window`. The kernel walks the sink/strided-middle/window kept set in place with the native `kv_stride`; no index build, no gather, no scratch.
- Defaults are the paper values: `sink = 4`, `window = 512`, `stride = 4` (`include/tge/models/qwen2.hpp`, `SparseConfig`). At 8,193 positions the kept count is `4 + (8193-516)/4 + 512 ≈ 2,435` (~30%); at 2,049 it is ~899 (~44%). The kept fraction falls with context, which is why the win grows.
- `attention_batch` (used by the prefill chunks and the speculative verifier) takes the dense block kernel only when `!sparse_.enabled` (`src/models/qwen2.cpp:1584`). With sparse enabled it falls back to the per-token path, so **the six verify rows also run the strided sparse kernel** — the second finding below.
- The killed 8k test ran before the port: `build_sparse_index` + compacted buffers copied ~2.4k rows per head per layer per token, and the copy bytes grew with context while the dot saving saturated. The strided kernel removes exactly that term.

## 2. Results

### Plain decode at 8k (interleaved, base/sparse order then sparse/base)

| pair | dense t/s | sparse t/s | ratio | dense attn | sparse attn |
|---|---:|---:|---:|---|---|
| 1 (dense -> sparse) | 5.40 | 9.52 | **1.76** | 1508.2 ms (54%) | 399.6 ms (25%) |
| 2 (sparse -> dense) | 7.04 | 9.57 | **1.36** | 1150.4 ms (54%) | 376.9 ms (24%) |

The dense arm drifted upward between pairs (5.40 -> 7.04 t/s; its attention phase moved 1508 -> 1150 ms), so the honest ratio range is **1.36-1.77x** with the sparse arm at 9.52/9.57 (0.5% spread). Even the slowest sparse run beats the fastest dense run by 1.36x. The attention phase alone is 3.0-3.8x faster; matvec is context-insensitive and stays ~1.14 s, so the end-to-end gain is bounded by the attention share, as expected.

### 2k control (same model/prompt family, n=32)

| arm | t/s | attn share |
|---|---:|---|
| dense | 11.39 / 11.78 | 16% / 15% |
| sparse | 12.19 / 12.43 | 10% / 11% |

Ratios 1.07 / 1.06. The near-2k win is smaller than the 8k win on this model, which is the kept-fraction trend (44% kept at 2k vs 30% at 8k) plus the smaller attention share.

### Speculation + sparse at 8k

| pair | spec t/s | spec+sparse t/s | ratio | spec line (spec -> spec+sparse) | attn share |
|---|---:|---:|---:|---|---|
| 1 (spec -> spec+sparse) | 9.68 | 15.61 | **1.61** | 4 steps, 3.00/step, 16 emitted -> 3 steps, 5.33/step, 18 emitted | 52% -> 34% |
| 2 (spec+sparse -> spec) | 9.37 | 16.36 | **1.75** | 4 steps, 3.00/step -> 3 steps, 5.33/step | 53% -> 35% |

Two mechanisms compound:

1. **The verify rows are sparse too.** With `--sparse`, `attention_batch`'s block path is gated off, so `verify_draft`'s six rows run the per-token strided kernel instead of the dense 128-query block. The verify's attention share falls from 52-53% to 34-35% of decode.
2. **Acceptance changes** (3.00 -> 5.33 accepted/step) because the sparse verifier's logits are the priced lossy tier's logits, not the dense ones. This is not a free gain: it is the same KL 2.6e-2-class subset already shipped behind `--sparse`, and the periodic prompt happens to accept more of its drafts under it.

Against the same session's dense plain decode (5.40-7.04 t/s), the combined arm is **2.2-3.0x** at 8k. Against the 8k spec-only arm it is 1.6-1.75x.

### Id parity (2k periodic control)

```text
dense : 578 4062 14198 39935 35308 927 279 16053 5679 13 578 4062 14198 39935 35308 927
sparse: 578 4062 14198 39935 35308 927 279 16053 5679 13 220 578 4062 14198 39935 35308
```

One near-tie flip at position 10, then re-convergence to the dense periodic sequence. This is the expected behaviour of the accepted sparse tier (fidelity KL 2.6e-2, top-1 975/1023 on the 1023-position corpus); no id-parity claim is made for sparse.

## 3. Falsifier (stated before the runs)

**REOPEN** the long-context sparse gate if sparse >= 1.10x dense at 8k with the attention phase materially faster; **KEEP the 2k-only gate** otherwise. Measured 1.36-1.77x plain and 1.61-1.75x on top of speculation: the gate reopens with margin. The 8k kill's own reopen condition (no-gather decode at long context) is met, and the kill remains correct for the gathered path it measured.

## 4. Reading and usage guidance

- **Long-context decode has a new first-order lever**: on the 3B at 8k, `--sparse` cuts decode attention by 3-3.8x and takes plain decode from 5.4-7.0 to 9.5 t/s. This is the largest single 8k decode change in the library since the f16 KV tier, and it composes with f16 (both arms here are f16).
- **Gate text update**: sparse decode now wins at ~2k and at 8k on the 3B with f16 KV; the recorded "wins near 2k, loses at 8k" gate was a property of the gathered kernel. i8 KV stays excluded (no strided i8 kernel), and the subset is lossy — the opt-in contract is unchanged.
- **Speculation + sparse is a real long-context pairing**: 15.6-16.4 t/s vs 9.4-9.7 spec-only at 8k. The mechanism is specific: the sparse flag routes the verifier off block attention onto per-token sparse attention, and the acceptance shift belongs to the priced tier. Do not read the 5.33/step as a sparse-drafter improvement that would carry to other prompts; it is a property of this periodic workload's logits under the subset.
- **At short context keep the existing advice**: attention is 15-16% at 2k on the 3B, so the plain-decode win is 1.05-1.07x; the tier pays mainly at long context, and at 8k it is decisive.

## 5. File anchors (current HEAD, read-before-write verified)

- `src/cpu/avx2/attention_avx2.cpp:263` — `attention_head_static_sparse_avx2` (strided f32).
- `src/cpu/avx2/attention_avx2.cpp:367` — `attention_head_static_sparse_fp16_avx2` (strided f16).
- `src/models/qwen2.cpp:1329-1342` — sparse dispatch in `attention_head`; gate `sparse_.enabled && !kv_q8_ && attention_mode != Scalar && span > sink + window`.
- `src/models/qwen2.cpp:1584` — `attention_batch`'s dense block gate (`!sparse_.enabled`), the reason the verifier goes per-token under sparse.
- `include/tge/models/qwen2.hpp` — `SparseConfig{sink=4, window=512, stride=4}`.
- `apps/cli/main.cpp:2177-2180, 2233-2236` — `--sparse`, `--sparse-window`, `--sparse-stride`.
- `reports/idea-invented-sparse-8k-killed.md` — the gathered-path kill this cycle reverses.
- `reports/winner-port-sparse-nogather-keep.md`, `winner-port-sparse-f16-keep.md` — the shipped kernels.

## 6. Reimplementation recipe

1. No code: use the shipped flags on the reference 3B, f16 KV, 10 decode threads, GPU off.
2. Build the compact 8k prompt (the fox text does not fit the Windows command line):

   ```powershell
   $p = ("The fox runs. " * 2048)      # 8,193 tokens
   ```

3. Plain arms, interleaved:

   ```powershell
   tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt $p --context 8704 --n 16 --kv f16 --threads 10 --gpu off [--sparse]
   ```

   Expect dense 5.4-7.0 t/s (attn 54%), sparse 9.5-9.6 t/s (attn 24-25%), ratio 1.36-1.77x; each run costs ~230-240 s of prefill.
4. Spec arms:

   ```powershell
   tge generate ... --speculative [--sparse]
   ```

   Expect spec 9.4-9.7 t/s (4 steps, 3.00/step, attn 52-53%), spec+sparse 15.6-16.4 t/s (3 steps, 5.33/step, attn 34-35%), ratio 1.61-1.75x.
5. 2k control: `fox x205`, `--context 2560 --n 32`, dense 11.4-11.8 vs sparse 12.2-12.4 (1.05-1.07x), attn 16% -> 10%.
6. Id check: add `--print-ids`; expect one near-tie divergence then re-convergence on periodic text. No code ships; sparse is already shipped and opt-in.

## 7. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56) from the cycle gate
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed

2k control (fox x205, ctx 2560, n 32):
  dense 11.39/11.78 t/s (attn 16/15%) ; sparse 12.19/12.43 t/s (attn 10/11%)
8k plain (compact fox x2048, ctx 8704, n 16):
  dense 5.40/7.04 t/s (attn 54%, 1508/1150 ms) ; sparse 9.52/9.57 t/s (attn 24-25%, 377-400 ms)
8k spec:
  spec 9.68/9.37 t/s (4 steps, 3.00/step, attn 52-53%)
  spec+sparse 15.61/16.36 t/s (3 steps, 5.33/step, attn 34-35%)
2k ids:
  dense  ... 13 578 4062 ... ; sparse ... 13 220 578 4062 ...   (one flip, re-converges)
git status --short                                -> clean (no code touched)
python aikit.py check aegean-tge                  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
