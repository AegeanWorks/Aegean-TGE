# MEASURED: 8k decode on non-periodic code — spec alone is only 1.13x, sparse is the robust lever (1.61x), and spec+sparse stacks to 1.71-1.83x

- **Idea**: Every 8k decode result in the library (`idea-invented-spec-at-8k-measured.md`, `idea-invented-sparse-8k-reopened.md`, `idea-invented-sparse-prefill-8k-measured.md`) used a periodic prompt where the drafter accepts 3-5 tokens/step. That is the best case for speculation. The decision-relevant long-context workload is non-periodic: code/template text with moderate prompt echo. Measure the full arm matrix (off, sparse, spec, spec+sparse) at 8k on repetitive C-like code to test whether the periodic ratios carry, and whether sparse and speculation still compose when acceptance is weaker.
- **Cycle date**: 2026-11-12
- **Verdict**: **MEASURED — the periodic 8k spec ratio does not carry; sparse does, and the composition still wins.** Llama-3.2-3B-Instruct-Q4_K_M, 8,104-token code prompt (213 C functions + unique tail), `--context 8704 --n 32 --kv f16 --threads 10 --gpu off`:
  - **off**: **6.52 t/s**; decode `matvec 46%, attn 52%`; prefill 211.3 s (attn 68.7 s).
  - **sparse**: **10.49 t/s (1.61x off)**; decode `matvec 70%, attn 28%` (attention 2486.6 -> 822.8 ms); prefill 175.5 s (attn 40.4 s, 0.59x).
  - **spec**: **7.34 / 7.46 t/s (1.13-1.14x off)**; `8 steps, 24 accepted (3.00/step), 31 emitted`; decode attn stays 53-54%. The 6-row verify at 8k costs almost as much as it saves when the drafter accepts 3/step.
  - **spec+sparse**: **11.12 / 11.92 t/s (1.71-1.83x off, 1.51-1.62x spec, 1.06-1.14x sparse alone)**; `7 steps, 19 accepted (2.71/step)`; decode attn 33-36%. Sparse lowers acceptance slightly on code (3.00 -> 2.71/step) yet still wins big: the money is in the verify attention, not the drafting.
  - **Ids**: sparse-only and spec+sparse are byte-identical on a 16-token check (`198 396 282 11584 1577 865 8 314 471 865 489 220 22 26 457 396`) — greedy speculation stays verifier-exact on top of the priced sparse tier.
- **Tree state at close**: no code touched (all flags shipped); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Llama-3.2-3B-Instruct-Q4_K_M.gguf |
| prompt | 213 x `int f$i(int x) { int y = x * $i + 7; if (y > 100) { y = y - 100; } return y; }` + `// unique tail 8f3a2c` |
| prompt size | 8,104 tokens, 17,271 characters (fits the Windows command line; prose at 8k does not) |
| context / n / KV / threads | 8704 / 32 / f16 / 10 |
| arms | off, `--sparse`, `--speculative`, `--speculative --sparse` |
| runs | off 1, sparse 1, spec 2, spec+sparse 2 (arms interleaved: off, spec, spec+sparse, spec+sparse, spec) |

The prompt is repetitive by construction so the prompt-lookup drafter has real material — this is a code/template workload, not natural prose. The earlier 1,530-token version of this shape measured 4.33 accepted/step and 1.88x spec; at 8k the drafter's reach is similar (3.00/step) but the verify cost is four times larger.

## 2. Results

### Decode, 8k code

| arm | t/s (runs) | median | ratio vs off | ratio vs spec | spec line | decode attn |
|---|---|---:|---:|---:|---|---:|
| off | 6.52 | 6.52 | 1.00 | — | — | 52% |
| sparse | 10.49 | 10.49 | **1.61** | — | — | 28% |
| spec | 7.34 / 7.46 | 7.40 | **1.13** | 1.00 | 8 steps, 24 acc (3.00/step), 31 emitted | 53-54% |
| spec+sparse | 11.12 / 11.92 | 11.52 | **1.77** | **1.56** | 7 steps, 19 acc (2.71/step), 26 emitted | 33-36% |

### Prefill, 8k code

| arm | total prefill | prefill attn | matvec | ratio vs off |
|---|---:|---:|---:|---:|
| off | 211.3 s | 68.7 s | 131.1 s | 1.00 |
| spec | 205.7-210.6 s | ~66.4-68.3 s | 128.0-130.8 s | 1.00-1.03 |
| sparse | 175.5 s | 40.4 s | 123.8 s | **1.20** |
| spec+sparse | 177.9-191.7 s | 39.5-54.8 s | 125.7-127.1 s | **1.10-1.19** |

Prefill behaves exactly like the periodic 8k result: sparse is 1.10-1.20x total with attention 0.58-0.80x, independent of workload.

### Ids (n=16, `--print-ids`)

```text
sparse      : 198 396 282 11584 1577 865 8 314 471 865 489 220 22 26 457 396
spec+sparse : 198 396 282 11584 1577 865 8 314 471 865 489 220 22 26 457 396
```

Byte-identical. Greedy speculation is verifier-exact; enabling it on top of sparse does not change a single token on this workload.

## 3. Reading

- **Speculation's 8k win is workload-shaped, not context-shaped.** On the periodic 8k prompt spec is 2.2-3.0x because 5.33 drafts accept per step; on code it accepts 3.00/step and the six-row verify's attention (53-54% of decode) consumes the gain, leaving 1.13x. The 8k periodic number should not be quoted as a general 8k expectation.
- **Sparse is the workload-robust 8k lever.** 1.61x on code vs 1.36-1.77x on periodic; attention 3.0x faster (52% -> 28% of decode) either way. The mechanism is content-blind, so the ratio tracks the kept fraction and box state, not the prompt.
- **The composition is driven by verify attention.** spec+sparse is 1.56x over spec on code even though sparse *reduces* acceptance (3.00 -> 2.71/step). With sparse, the verify rows run the per-token sparse kernel (the shipped fallback), so the six-row attention is ~2.5x cheaper; that is what pays. The block-sparse prefill recipe (`idea-invented-block-sparse-prefill-keep.md`) would improve the prefill side further but is not in this measurement (spike reverted; shipped tree is per-token).
- **Against sparse alone, speculation adds only 6-14% at 8k on code.** Usage: at 8k on non-periodic work, `--sparse` is the lever; `--speculative` is a small add-on when the prompt echoes (code templates do echo, so it is worth it here), and the combination is the best measured 8k CPU configuration on this box: 1.71-1.83x off.
- **Prize the prefill side too**: at 16-32 generated tokens the run is prefill-dominated (~180-210 s vs 2.6-4.8 s decode), so the end-to-end gain from either flag is the prefill ratio; the decode ratios above are for long generations.

## 4. Falsifier (stated before the runs)

**MEASURED/CONFIRMED** if spec+sparse >= 1.3x spec on the 8k code prompt with identical sparse-vs-spec+sparse ids, and sparse >= 1.3x off. **KILL the composition** if sparse makes spec slower on code (the tier's logits could have hurt acceptance enough to lose). Measured 1.51-1.62x over spec and 1.71-1.83x over off with byte-identical ids: the composition passes. The spec-alone ratio (1.13x) is recorded as the workload finding, not a falsifier failure.

## 5. File anchors (current HEAD, read-before-write verified)

- `apps/cli/main.cpp:788-922` — fused greedy chain (6-row verify, sparse-affected logits).
- `apps/cli/main.cpp:924-1070` — non-fused greedy path and the correction forwards that dominate the code workload (3.00/step, many partial accepts).
- `src/models/qwen2.cpp:1329-1342` — per-token sparse dispatch used by decode and by the verifier when `--sparse` is on.
- `src/models/qwen2.cpp:1580-1615` — `attention_batch` dense block gate (`!sparse_.enabled`), the reason the sparse verifier is per-token on the shipped tree.
- `include/tge/models/qwen2.hpp` — `SparseConfig{sink=4, window=512, stride=4}`.
- `reports/idea-invented-spec-at-8k-measured.md` — the periodic 8k spec table this cycle bounds.
- `reports/idea-invented-sparse-8k-reopened.md` — the periodic 8k sparse reopening.
- `reports/idea-invented-block-sparse-prefill-keep.md` — the prefill kernel recipe not applied here.

## 6. Reimplementation recipe

1. Build the code prompt in PowerShell (17.3k chars; fits `Start-Process` as one quoted argument):

   ```powershell
   $sb = New-Object System.Text.StringBuilder
   foreach ($i in 1..213) {
     [void]$sb.AppendLine("int f$i(int x) { int y = x * $i + 7; if (y > 100) { y = y - 100; } return y; }")
   }
   [void]$sb.Append("// unique tail 8f3a2c")
   $p = $sb.ToString()      # 8,104 tokens
   ```

   Confirm with `--context 64 --n 1`: the CLI prints `prompt : 8104 tokens from text` before the fit error.

2. Run the arms with `Start-Process` redirection (long prompt through the pipeline truncates):

   ```powershell
   tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt $p --context 8704 --n 32 --kv f16 --threads 10 --gpu off [--sparse] [--speculative]
   ```

3. Expect off ~6.5 t/s (attn 52%), sparse ~10.5 t/s (attn 28%), spec ~7.3-7.5 t/s (3.00/step, 8 steps), spec+sparse ~11.1-11.9 t/s (2.71/step, 7 steps); prefill 175-211 s depending on arm. Each run costs ~3-4 min of prefill.
4. Id parity: rerun sparse and spec+sparse with `--n 16 --print-ids`; the id strings must match exactly.
5. No code ships.

## 7. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56, from the cycle gate)
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed

code prompt build -> 17,271 chars / 8,104 tokens (ctx 64 count check)
8k code, ctx 8704, n 32, f16, 10 threads:
  off          : 6.52 t/s ; decode attn 52% ; prefill 211.3 s (attn 68.7)
  sparse       : 10.49 t/s (1.61x) ; attn 28% ; prefill 175.5 s (attn 40.4, 0.59x)
  spec         : 7.34 / 7.46 t/s (1.13-1.14x) ; 8 steps, 24 acc (3.00/step) ; prefill 205.7-210.6 s
  spec+sparse  : 11.12 / 11.92 t/s (1.71-1.83x off, 1.51-1.62x spec) ; 7 steps, 19 acc (2.71/step) ; prefill 177.9-191.7 s
ids (n=16):
  sparse      : 198 396 282 11584 1577 865 8 314 471 865 489 220 22 26 457 396
  spec+sparse : 198 396 282 11584 1577 865 8 314 471 865 489 220 22 26 457 396
git status --short                                -> clean (no code touched)
python aikit.py check aegean-tge                  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
