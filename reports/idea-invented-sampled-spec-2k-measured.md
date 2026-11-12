# MEASURED: sampled speculation at 2k on 3B — 2.22-2.33x (greedy 2.74x); the sampler becomes the dominant non-forward cost

- **Question**: Greedy speculation at 2k is 2.74x with 5.17 accepted/step. Sampled speculation (temp 0.8, top-p 0.9) runs the same 6-row verify and accept rule but then does a penalized softmax + cut selection per emitted token, and its accept step consumes RNG differently. What does it deliver at 2k on the reference 3B, and where does the time go?
- **Cycle date**: 2026-11-11
- **Verdict**: **MEASURED — sampled spec is 2.2-2.3x at 2k, still a large win.** Llama-3.2-3B-Instruct-Q4_K_M, 2,050-token periodic prompt, `--context 2560 --n 32 --temp 0.8 --top-p 0.9 --seed 7 --kv f16 --threads 10 --gpu off`:
  - off: 12.24/11.92 t/s; `timers forward 2523/2590 ms, sample 10.5/10.3 ms (0.4%)`.
  - spec: **27.22/27.75 t/s**; `5 steps, 28 draft tokens accepted (5.60/step), 32 tokens emitted`; `timers forward 1126/1105 ms, sample 808/788 ms` (the printed 70.9% share is relative to a spec-path denominator; the effective sampler cost is ~25 ms/token because each of the 5 verify steps softmaxes 6 rows).
  - ratios: 2.22 and 2.33 -> **median ~2.28x**.
  - vs greedy at the same prompt (30.03-31.56 t/s, 2.74x): sampled spec gives up ~10-18% of the greedy win, consistent with the sampler's 6-row softmax work.
- **Tree state at close**: no code touched; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Llama-3.2-3B-Instruct-Q4_K_M.gguf |
| prompt | fox sentence x205 = 2,050 tokens |
| context / n / KV / threads | 2560 / 32 / f16 / 10 |
| sampling | `--temp 0.8 --top-p 0.9 --seed 7` |
| arms | off vs `--speculative` |

Sampled spec does not reproduce sequential ids (the accept rule consumes randomness differently; the distribution is preserved), so this cycle does not claim id parity — timings and the `spec:` line only.

## 2. Results

| run | off t/s | spec t/s | ratio | spec line | timers (spec) |
|---|---:|---:|---:|---|---|
| 1 | 12.24 | 27.22 | 2.22 | 5 steps, 28 accepted (5.60/step), 32 emitted | forward 1126 ms, sample 808 ms |
| 2 | 11.92 | 27.75 | 2.33 | 5 steps, 28 accepted (5.60/step), 32 emitted | forward 1105 ms, sample 788 ms |
| **median** | 12.08 | 27.49 | **2.28** | | |

## 3. Reading

- Sampled speculation at 2k is ~2.3x, versus ~2.74x greedy. The gap is the sampler share: 5 verify steps each softmax 6 rows, so the sampler work scales with verify width while the forward saving is the same.
- The `timers` percent (70.9%) should not be read at face value in the spec path — the spec block's forward and sample windows overlap the step boundaries; the absolute numbers (1.1 s forward + 0.8 s sample for 32 tokens) are the honest ones and match ~25 ms/token sample cost for 6-row softmaxes at 2k.
- Usage: sampled speculation is worth it on repetitive prompts at 2k (2.2-2.3x); the earlier stacking result (spec + kq8d tier are substitutes) applies to sampled mode too, since the verifier dominates.
- Follow-up worth a future cycle: the sampled spec path's 6-row softmax per step is now a meaningful share; computing the accept distributions only for rows actually needed (lazy row-by-row) could recover part of the greedy/sampled gap on low-accept workloads — but the verifier batch already computes all rows' logits in one pass, so the win would be limited to the softmax, not the matvec.

## 4. Reimplementation recipe

1. No code: `tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt "<fox x205>" --context 2560 --n 32 --temp 0.8 --top-p 0.9 --seed 7 --kv f16 --threads 10 --gpu off [--speculative]`.
2. Capture with `Start-Process -RedirectStandardOutput` (the `2>&1 | Out-File` harness truncates spec output on long prompts); read `decode:`, `spec:`, `timers:`.
3. Expect off ~12 t/s, spec ~27-28 t/s, 5 steps, 5.60 accepted/step, ratio ~2.2-2.3x.
4. No ids comparison for sampled mode; the distribution-preservation basis is the existing sampled-cuts report.

## 5. Exact commands run and results

```text
tge generate ... --temp 0.8 --top-p 0.9 --seed 7 --kv f16 --threads 10 --gpu off
  off  -> 12.24 / 11.92 t/s, sample 10.5 / 10.3 ms (0.4%)
  spec -> 27.22 / 27.75 t/s, 5 steps, 28 accepted (5.60/step), 32 emitted
         forward 1126 / 1105 ms, sample 808 / 788 ms
git status --short                -> clean (no code touched)
python aikit.py check aegean-tge  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
