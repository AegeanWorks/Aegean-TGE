# CORRECTION: i8 KV quality is corpus-dependent — 99.75% on repeated text vs 90.05% on prose at the same 2k context; the recorded 2k bar was corpus-flattered

- **Idea**: `idea-invented-kv-tier-quality-8k-measured.md` compared i8 against its recorded 2k bar (99.9%, KL 1.303e-3) and concluded the tier "does not hold at 8k". The 2k bar itself came from `idea-invented-i8-longctx-confirmed.md`, which used a **repeated-paragraph** corpus. Top-1 agreement is a near-tie statistic, so a trivial corpus can hide flips. Measure the same tier at the same 2k context on two corpora — repeated paragraph vs natural prose (`architecture.md`) — with an f16 control, and re-read the 8k ladder in that light.
- **Cycle date**: 2026-11-13
- **Verdict**: **CORRECTED — i8's quality cost is set by the text's near-tie density, not by context alone, and the recorded 2k bar describes only repetitive text.** Qwen2.5-0.5B-Instruct-Q8_0, `--context 2048 --tokens 2000 --kv {f16|i8} --threads 10 --gpu off`, 1,999 teacher-forced positions, candidate vs full-scalar f32-KV oracle:
  - **i8, repeated-paragraph corpus**: 1994/1999 (99.75%), KL 1.226e-03 — reproduces the recorded 99.9% / 1.303e-3.
  - **i8, natural prose corpus** (`architecture.md` x9 slice): **1800/1999 (90.05%)**, KL 2.828e-02 — a **9.70pp** corpus effect at identical context, model and tier.
  - **f16, natural prose corpus**: **1998/1999 (99.95%)**, KL 2.142e-06 — the harness is clean; the prose arm is not noise.
  - **The recorded 2k i8 bar (98.6% dossier / 99.9% measured) is a best-case number for repetitive text.** On prose the tier loses ~10pp at 2k, not 0.1pp.
  - **Consequence for the 8k ladder**: the f16-vs-i8 *difference* at 8k was measured on the same corpus (99.98% vs 96.60%) and stands as a valid tier comparison; but the absolute i8 figures at 2k/3k/4k/8k on the `architecture.md x N` family are inflated by prompt repetition (the model sees the document once per copy), so they must not be quoted as natural-text quality or as a context curve.
- **Tree state at close**: no code touched (shipped flags only); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Qwen2.5-0.5B-Instruct-Q8_0 |
| corpora | (a) repeated paragraph: first 136-char paragraph of `architecture.md` x100 (13,802 bytes, 1,999 positions) |
| | (b) natural prose: `architecture.md` x9 (107,624 bytes), first 1,999 positions scored |
| context / tokens | 2048 / 2000 |
| threads / GPU | 10 / off |
| harness | candidate tier + production kernels vs full-scalar f32-KV oracle (`tge fidelity`) |

The two 2k arms differ only in corpus text; model, context, tier, threads and harness are identical. The f16 prose arm isolates harness/first-copy effects from tier effects.

## 2. Results

### The paired 2k comparison

| arm | corpus | mean KL | worst KL | max |diff| | top-1 | verdict |
|---|---|---:|---:|---:|---:|---|
| i8 | repeated paragraph | 1.226e-03 | 3.009e-01 | 1.673e+01 | **1994/1999 (99.75%)** | reproduces the 2k record |
| i8 | natural prose | **2.828e-02** | 3.103e+00 | 9.751e+00 | **1800/1999 (90.05%)** | **−9.70pp corpus effect** |
| f16 | natural prose | 2.142e-06 | 1.295e-04 | 5.857e-02 | **1998/1999 (99.95%)** | control: harness clean |

Same tier, same context, same model: the corpus moves top-1 by 9.7pp and mean KL by 23x. The tier's error does not depend on quantizer arithmetic changing with text; it depends on how many positions sit near a top-1 tie, where the i8 rounding can flip the argmax.

### The `architecture.md` family readings (context × repetition confound)

| context | positions | i8 top-1 | i8 mean KL | f16 top-1 (same corpus) | note |
|---:|---:|---:|---:|---:|---|
| 2,048 | 1,999 | 1800 (90.05%) | 2.828e-02 | 1998 (99.95%) | first copy of the document |
| 3,584 | 3,399 | 3131 (92.12%) | 2.455e-02 | — | first ~1.7 copies |
| 4,608 | 4,399 | 4128 (93.84%) | 2.288e-02 | — | first ~2.2 copies |
| 8,704 | 8,199 | 7920 (96.60%) | 1.657e-02 | 8197 (99.98%) | all 9 copies |

The apparent "i8 improves with context" is a **repetition artifact**: the corpus repeats the same document, and later positions are easier for any tier. The correct comparison is same-corpus f16 vs i8 at each length: the tier gap is ~10pp at 2k and ~3.4pp at 8k on this family — the gap shrinks because the text gets easier, not because i8 gets better.

## 3. Reading

- **Top-1 agreement is a corpus property.** A repeated-paragraph corpus is nearly deterministic for a 0.5B model; it cannot price a near-tie tier. The i8 bar should be quoted per corpus class, and the prose number (90.05% at 2k) is the one that matters for chat/RAG/code use.
- **i8 is a capacity tier with a ~10pp prose cost at 2k.** Its KV-quartering and the 1.84x continuous-batching admission win (`idea-invented-batchsnap-kindtag-measured.md`, `measurement-log` 2026-11-11) remain real; the quality claim attached to it must be restated. The dossier bar (KL 1e-3, top-1 98.6%) was met only on repetitive text.
- **The 8k tier ladder needs the same caveat.** `idea-invented-kv-tier-quality-8k-measured.md` compared f16 (99.98%) and i8 (96.60%) on the same `architecture.md x9` corpus: that comparison is valid, but both absolute numbers are repetition-flattered. A natural-prose 8k reading would require a non-repeating 8k corpus (e.g. concatenating distinct docs to 8k tokens); that is the clean next measurement for the ladder.
- **The speed claims are untouched**: i8's 8k decode is a measured loss (`idea-invented-i8-8k-killed.md`), its 2k decode gain and the continuous-batching admission win are measured elsewhere. Only the quality framing changes.
- **Recommendation update**: do not quote "i8 holds 99.9% at 2k" for general text; quote 90.05% at 2k on prose, and use i8 only when the memory saving is worth ~10pp top-1 or the workload is repetitive enough to be near-tie-free.

## 4. Falsifier (stated for this diagnostic)

**CONFIRM the recorded bar** if i8 prose top-1 at 2k is within 1pp of the repeated-paragraph result. **CORRECT** if the gap is larger. Measured 9.70pp with the f16 control at 99.95%: correct. A harness bug is ruled out by the control; a model-file difference is ruled out by the identical path and model.

## 5. File anchors (current HEAD, read-before-write verified)

- `apps/cli/main.cpp:2520-2545` — fidelity reference (f32 KV, scalar kernels) vs candidate tier; the comparison that makes this a tier price.
- `apps/cli/main.cpp:2740+` — `--kv` in fidelity.
- `src/models/qwen2.cpp:117-134` — `store_kv_q8` (per-32 quantization at store time).
- `src/models/qwen2.cpp:1467-1485` — q8 KV attention reads with per-32 scales.
- `reports/idea-invented-i8-longctx-confirmed.md` — the repeated-paragraph 2k bar this cycle re-prices.
- `reports/idea-invented-kv-tier-quality-8k-measured.md` — the 8k ladder whose absolute readings need this caveat.
- `reports/idea-invented-i8-8k-killed.md` — the 8k i8 speed loss (unchanged).

## 6. Reimplementation recipe

1. Repeated-paragraph corpus: first paragraph of `docs/architecture.md` with `Length > 100` chars, `($para + "`n`n") * 100` -> 13,802 bytes, 1,999 positions at ctx 2048.
2. Prose corpus: `docs/architecture.md` x 9 -> 107,624 bytes; slice the first 1,999 positions with `--context 2048 --tokens 2000`.
3. Runs (each ~74-75 s):

   ```text
   tge fidelity --model qwen2.5-0.5b-instruct-q8_0.gguf --corpus <repeat|prose8k> \
     --context 2048 --tokens 2000 --kv {i8|f16} --threads 10 --gpu off
   ```

   Expect i8 repeat 1994/1999 (KL 1.226e-03), i8 prose 1800/1999 (2.828e-02), f16 prose 1998/1999 (2.142e-06).
4. Do not use a self-repeating corpus to price a near-tie tier; use distinct documents concatenated to the target length.
5. No code ships.

## 7. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56, cycle gate)
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed

fidelity (0.5B Q8_0, ctx 2048, 2000 tokens, 10 threads, gpu off):
  i8, repeated paragraph -> KL 1.226e-03, worst 3.009e-01, max|diff| 1.673e+01, 1994/1999, 73.6 s
  i8, architecture prose -> KL 2.828e-02, worst 3.103e+00, max|diff| 9.751e+00, 1800/1999, 74.1 s
  f16, architecture prose-> KL 2.142e-06, worst 1.295e-04, max|diff| 5.857e-02, 1998/1999, 75.2 s

context sweep on the arch x9 family (i8): 3399 pos 92.12% KL 2.455e-02; 4399 pos 93.84% KL 2.288e-02;
  8199 pos 96.60% KL 1.657e-02 (repetition-flattered; do not read as a context curve)
git status --short                                -> clean (no code touched)
python aikit.py check aegean-tge                  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
