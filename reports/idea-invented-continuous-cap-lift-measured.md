# INVENTED: lift the continuous-bench prompt cap from 64 to the real 128 prefill rows — MEASURED (safe; kq8 win 1.55-1.67x at 96/120-token prompts)

- **Idea**: `bench continuous` rejects `prompt_tokens + sequences > 64` with "must fit the 64 prefill rows", but the batch buffers are sized from `kPrefillChunk = 128` (`src/models/qwen2.cpp:38`, `prefill_rows = min(kPrefillChunk, context)` at `:624-626`). The 64-row guard is stale and blocks the one measurement the harness could not make: continuous batching with realistic (64-120-token) prompts. Lift the guard to 128, then re-price the path-invariant kq8 change at 96 and 120 prompt tokens.
- **Cycle date**: 2026-11-11
- **Verdict**: **MEASURED — the lifted cap works and the kq8 win holds.** Llama-3.2-3B-Instruct-Q4_K_M, `bench continuous --requests 8 --tokens 8 --threads 10 --sequences 8 --kv f16 --prompt-tokens N`:
  - **96 tokens**: float 12.61/12.31 -> kq8 19.51/19.55 t/s (**1.55/1.59x**).
  - **120 tokens**: float 9.41/8.86 -> kq8 14.79/14.81 t/s (**1.57/1.67x**).
  - At the old cap (56 tokens): 1.36-1.57x; the win does not decay with prompt length in this range.
  - **Primes engage at the larger admissions**: `POPULATE C=120` + 7 `PRIME C=120`; `state: independent (all transcripts identical)`.
- **Tree state at close**: spike reverted from `apps/cli/main.cpp` (guard back to 64) and `src/models/qwen2.cpp` (path-invariant flips reverted); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

- The guard at `apps/cli/main.cpp:1535` (`prompt_tokens + sequences > 64`) predates the 128-row prefill buffers; `kPrefillChunk` is 128 and `prefill_rows` is `min(128, context)`. Lifting the guard to 128 is a one-line spike (`sequences > 128 || prompt_tokens + sequences > 128` plus the message) and requires no buffer changes.
- The A/B used the path-invariant kq8 change (`tokens >= 1` + eight `float_only` flips) from `reports/idea-invented-continuous-kq8-path-invariant-keep.md` versus a float-batch baseline rebuilt with the same lifted cap (the old baseline binary rejects 96/120, so the baseline had to be rebuilt with only the cap change).

## 2. Results

| prompt tokens | float t/s | kq8 t/s | ratio |
|---:|---:|---:|---:|
| 56 (old cap) | 15.18/15.11 | 23.88/20.56 | 1.573/1.361 |
| 96 | 12.61/12.31 | 19.51/19.55 | 1.547/1.588 |
| 120 | 9.41/8.86 | 14.79/14.81 | 1.572/1.672 |

Primed run at 120 tokens: `POPULATE C=120 layers=28 kv_dim=1024 slot=0 (batch)` + `PRIME seq=1..7 C=120 slot=0`; `state: independent (all transcripts identical)`; one window read 17.07 t/s.

## 3. Reading

- The cap was an artificial harness limit, not a buffer constraint; lifting it is safe and lets the continuous bench exercise prompt lengths that match real chat/RAG loads. The kq8 path-invariant win is stable across 56/96/120 tokens (1.36-1.67x), with the ratio rising slightly at 120 as admission prefill and KV stores grow.
- Prime insertion at C=120 also exercises the larger memorised prefix (the table cap is 256 rows), and the transcripts stay independent.
- Promotion guidance: ship the guard lift on its own (harness-only, no numerics); keep the kq8 path-invariant change behind its flag/decision as before. With the cap lifted, the closeout's "long-prompt continuous impossible in this harness" caveat is removed for <=120-token prompts; 2k-prompt continuous still needs a larger batch buffer or a chunked admission path (new surface).
- No fidelity run owed: harness validation only; the A/B's kq8 arm is the already-priced near-exact path and its transcripts are identical.

## 4. Reimplementation recipe

1. `apps/cli/main.cpp:1535`: `prompt_tokens + sequences > 64` -> `> 128` (and the message "64" -> "128").
2. Build + 171/171. For the A/B, also apply the path-invariant kq8 change; build a float-batch baseline with only the cap lifted.
3. `bench continuous --model <3B q4_k_m> --requests 8 --tokens 8 --threads 10 --sequences 8 --kv f16 --prompt-tokens {56,96,120}`; expect kq8/float 1.36-1.67x, `state: independent`, and PRIME lines at the longer prompts.
4. Revert both; the guard lift is a one-line harness change if it is ever wanted permanently.

## 5. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge bench continuous ... --prompt-tokens {56,96,120} --sequences 8 --kv f16
  -> 1.36-1.67x kq8 vs float; POPULATE + PRIME C=120; state independent
git checkout -- apps/cli/main.cpp src/models/qwen2.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the rebuilt binary; the committed tree is unchanged except this report.
