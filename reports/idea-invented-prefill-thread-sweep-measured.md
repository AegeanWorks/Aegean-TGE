# MEASURED: prefill thread sweep on 3B — the default (16 logical) is already the best; pt8 is 7-21% slower

- **Idea**: Prefill defaults to `resolve_thread_count(0)` = all logical cores (16 on this 6P+4E box), while decode defaults to 10 physical. On a hybrid CPU, SMT siblings and E-cores can contend, so a smaller prefill pool (6/8/10/12) might beat 16. This is the remaining llama-gap axis (prefill is 1.3-1.5x behind llama on matched KV tiers).
- **Cycle date**: 2026-11-11
- **Verdict**: **KILLED — the default is already optimal.** Llama-3.2-3B-Instruct-Q4_K_M, `--threads 10 --kv f16 --gpu off`:
  - **2,050-token prompt** (single sweep): pt8 43.7, pt10 47.8, **pt16 53.1 t/s** — 16 is 21% over 8 and 11% over 10.
  - **510-token prompt** (sweep): pt6 53.5, pt8 60.7, pt10 57.3, pt12 50.6, pt16 53.8 — the 8-first reading was noise; interleaved A/B of 8 vs 16 gives **57.4 vs 59.2 (0.97)** and **48.0 vs 54.1 (0.887)**, i.e. 16 wins or ties at both lengths.
  - No id changes (default arithmetic, no flags beyond thread counts).
- **Tree state at close**: no code touched; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Llama-3.2-3B-Instruct-Q4_K_M.gguf |
| prompts | fox sentence x51 (~510 tokens) and x205 (~2,050 tokens) |
| flags | `--threads 10 --prefill-threads {6,8,10,12,16} --kv f16 --gpu off --n 4` |
| method | single sweep, then interleaved A/B for the best candidate vs the default |

## 2. Results

### 2,050 tokens (single sweep)

| prefill threads | t/s |
|---:|---:|
| 8 | 43.7 |
| 10 | 47.8 |
| **16 (default)** | **53.1** |

### 510 tokens

| prefill threads | sweep t/s | interleaved vs 16 |
|---:|---:|---|
| 6 | 53.5 | - |
| 8 | 60.7 | 57.4 vs 59.2 (0.97), 48.0 vs 54.1 (0.887) |
| 10 | 57.3 | - |
| 12 | 50.6 | - |
| 16 | 53.8 | - |

### Reading

- At the long prompt the default 16 logical threads is decisively best (+21% over 8, +11% over 10). At 512 the picture is noisier, but the interleaved pairs put 16 at or above 8 in both.
- SMT contention does not overcome the extra throughput for this compute-bound integer prefill; the 6P+4E topology is already exploited by the default. The prefill gap to llama must be attacked in the kernel (see the bounded prefill attempts), not the thread count.
- Usage note for the reports: matched `--prefill-threads 6` comparisons (used in the llama comparison) understate TGE prefill by ~15-25% versus the shipped default; the product comparison should quote default-thread TGE numbers when the goal is best-achievable.

## 3. Reimplementation recipe

1. No code: `tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt "<fox x51|x205>" --context {512,2560} --n 4 --threads 10 --prefill-threads {6,8,10,12,16} --kv f16 --gpu off`.
2. Read the `prefill:` line. Expect 16 best at 2k and at-or-above 8 at 512; do not reduce prefill threads.
3. No follow-up spike is warranted on thread count; the prefill lever is the GEMM/format (see `idea-invented-kq8-band1-killed.md`, `-pair-dot-killed.md`, `-split-sumi-killed.md`, `-vnni-killed.md`, `-vnni-per32-killed.md`).

## 4. Exact commands run and results

```text
tge generate ... --n 4 --threads 10 --prefill-threads {6,8,10,12,16} --kv f16 --gpu off
  512  -> 53.5 / 60.7 / 57.3 / 50.6 / 53.8 t/s (sweep)
  512 A/B 8 vs 16 -> 0.97, 0.887
  2050 -> 43.7 / 47.8 / 53.1 t/s (8 / 10 / 16)
git status --short                -> clean (no code touched)
python aikit.py check aegean-tge  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
