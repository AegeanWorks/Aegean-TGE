# MEASURED: speculative transcript equivalence sweep — KEEP with a kernel-path caveat (6/8 identical; exact only with matching kernels)

- **Idea**: Cycle 43 showed `--speculative` at ~2x on periodic prompts with identical ids. This cycle sweeps eight diverse prompts to test transcript equivalence, chases the divergences, and identifies the cause.
- **Cycle date**: 2026-11-08
- **Verdict**: **KEEP as a speed mode, with an equivalence caveat.** 6/8 short prompts produce **identical** ids to baseline; 2/8 diverge at late near-ties (tokens 36 and 44). The cause is **kernel-path mixing**, not a verifier bug: the speculative verifier uses the float-only batch path, while baseline single-token decode uses the kq8 integer path. **Baseline float vs baseline kq8 already differs** on the same prompts, and **baseline float == spec float on 2/2** divergent cases. Speculation is exact relative to the same kernel path, not to the default kq8 baseline. Code-like prompts are a major win: **4.66 → 23.23 t/s (4.98x)**.
- **Tree state**: measurement-only; no code change, no branch, no spike. 145/145 tests and 4/4 gates on the current tree; nothing pushed.

## 1. Mechanism and the equivalence question

`--speculative` is verifier-gated: accepted draft tokens must match the verifier's own argmax. But the verifier's batched forward routes the output projection and K-quant matvecs through `batch_matvec(..., float_only=true)` for occupancy-invariant math, whereas single-token baseline decode uses the kq8 integer kernels (default on). Two different arithmetics therefore produce the logits that decide token 36+; near-ties can flip.

## 2. Falsifier

- **KEEP as a usage mode** if the sweep is mostly identical and the speed gain is real.
- **Characterize and document** any divergence rather than calling it a bug.

## 3. Measured numbers

### Eight-prompt sweep, n=64, f32 KV

| # | prompt (abridged) | ids | baseline | spec | ratio | spec accepted |
|---:|---|---|---:|---:|---:|---:|
| 1 | quick brown fox / explain | IDENTICAL | 4.82 | 5.24 | 1.09x | 2.00/step |
| 2 | haiku about circuits | IDENTICAL | 4.82 | 4.70 | 0.98x | 1.00/step |
| 3 | compiler loop reasons | IDENTICAL | 4.72 | 4.97 | 1.05x | 1.67/step |
| 4 | **fibonacci code** | IDENTICAL | 4.66 | **23.23** | **4.98x** | **4.82/step** |
| 5 | **fisherman story** | **DIFF @ token 44** | 4.86 | 5.35 | 1.10x | 1.86/step |
| 6 | TCP vs UDP | IDENTICAL | 4.50 | 5.15 | 1.14x | 4.00/step |
| 7 | translate good morning | IDENTICAL | 4.92 | 5.31 | 1.08x | 2.00/step |
| 8 | **17 * 23 steps** | **DIFF @ token 36** | 4.90 | 6.55 | 1.34x | 1.93/step |

All runs emitted the full 64 tokens (no EOS). The two divergences are single late near-tie flips.

### Cause test on the two divergent prompts

| comparison | prompt 5 | prompt 8 |
|---|---|---|
| baseline default vs baseline `--kernel float` | DIFF | DIFF |
| baseline default vs spec default | DIFF | DIFF |
| baseline default vs spec `--kernel float` | DIFF | DIFF |
| **baseline `--kernel float` vs spec `--kernel float`** | **IDENTICAL** | **IDENTICAL** |

The default baseline already differs from the float baseline — the kq8 approximation is visible on these prompts independent of speculation. Once both arms use the same kernel path, speculation is exactly equivalent.

### Speed context

Cycle 43: periodic 4096 prompt → 1.95x / 2.15x. This sweep: code prompt **4.98x**, natural prompts 0.98–1.34x, averaging ~1.2x with a code/repetition tail.

## 4. Interpretation

- Speculation is a large speed lever for code, templates, and repetitive text, and a small/none lever for free prose.
- Its output is verifier-exact **for the arithmetic the verifier runs**. Under default kernels, the verifier (float batch) and sequential decode (kq8 integer) disagree on near-ties, so `--speculative` does not guarantee transcript equality with the default baseline. The historical "ids identical" results were true for their prompts but are not a general invariant.
- This is the same arithmetic split the `float_only` flag exists to manage for continuous batching; speculation exposes it at the token level.
- If exact equivalence to the kq8 baseline is required, speculation must not be enabled, or both modes must run `--kernel float` (which changes the baseline and may cost prefill speed).

## 5. Disposition

**KEEP as a speed mode, documented with the caveat.**
- Enable for code/template/repetitive workloads; expect 2–5x.
- Free-form prose: small or no gain.
- Transcript equivalence: exact only with matching kernel paths (`--kernel float` both arms). Under default kq8 decode, late near-tie flips are possible (2/8 short prompts here).

## 6. Exact commands

```text
.\build\release\bin\tge_tests.exe                      → 145/145
python aikit.py check aegean-tge                        → 4/4
tge generate --model Llama-3B --prompt "<p>" --n 64 --threads 6 \
  --gpu off --kv f32 --print-ids [--speculative] [--kernel float]
```

No push, no merge, no force-push. No code changed.
