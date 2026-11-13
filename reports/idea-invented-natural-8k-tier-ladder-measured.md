# MEASURED: natural-text 8k tier ladder — f16 99.90%, i8 91.33%, sparse 86.55%/80.81%; the self-repeating corpus flattered i8 and penalised sparse

- **Idea**: The previous correction (`idea-invented-i8-corpus-dependence-correction.md`) showed that i8's recorded 2k bar came from a self-repeating paragraph corpus, and named the clean next measurement: a non-repeating 8k corpus built from distinct documents, to re-price the whole tier ladder without repetition flattery. `idea-invented-kv-tier-quality-8k-measured.md` and `idea-invented-sparse-8k-quality-correction.md` used `architecture.md` x N, whose scored 8,199-token slice contains the same document three times. Measure f16, i8, and sparse (stride 2 and 4) at 8,199 positions on a corpus of eight distinct project documents.
- **Cycle date**: 2026-11-13
- **Verdict**: **MEASURED — on realistic text the ladder is f16 99.90% > i8 91.33% > sparse/2 86.55% > sparse/4 80.81%, and only f16 is quality-preserving.** Qwen2.5-0.5B-Instruct-Q8_0, 8,199 teacher-forced positions, `--context 8704 --tokens 8200 --kv f16 --threads 10 --gpu off`, candidate vs full-scalar f32-KV oracle:
  - **f16**: KL **2.312e-06**, worst 1.054e-03, max |diff| 8.051e-02, top-1 **8191/8199 (99.90%)**.
  - **i8**: KL 3.054e-02, worst 5.850e+00, max |diff| 1.274e+01, top-1 **7488/8199 (91.33%)** — **−8.57pp**, below its 98.6% bar.
  - **sparse 512/2** (~53% kept): KL 1.376e-01, top-1 **7096/8199 (86.55%)** — **−13.35pp**.
  - **sparse 512/4 default** (~30% kept): KL 2.573e-01, top-1 **6626/8199 (80.81%)** — **−19.09pp**.
  - **The repetition corpus was wrong in both directions**: it reported i8 at 96.60% (natural: 91.33%, so it **flattered i8 by 5.27pp**) and sparse/4 at 67.49% (natural: 80.81%, so it **penalised sparse by 13.32pp**). The tier ordering it implied (i8 nearly free, sparse catastrophic) is directionally right but the magnitudes belonged to that corpus, not to text.
  - **Natural-text context behaviour differs by tier**: f16 99.95% (2k) -> 99.90% (8k) holds; i8 90.05% (2k) -> 91.33% (8k) is **flat, not compounding**; sparse/4 88.85% (2k) -> 80.81% (8k) **does compound** (−8.04pp as context quadruples). The earlier "i8 error compounds" reading was itself an artifact of the repeating corpus improving with length.
- **Tree state at close**: no code touched (shipped flags only); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Qwen2.5-0.5B-Instruct-Q8_0 |
| corpus | eight distinct project docs concatenated: `architecture.md` (11,957 B), `report-0.1.md` (14,553), `report-0.2-verification.md` (19,180), `benchmarks.md` (5,114), `roadmap.md` (5,766), `hardware-support.md` (4,189), `fused-correction.md` (30,529), `README.md` (17,522) = **108,826 B**, ~27k tokens |
| scored slice | first **8,199 positions** (architecture + report-0.1 + part of report-0.2 — three distinct documents, no repetition) |
| context / tokens | 8704 / 8200 |
| threads / GPU | 10 / off |
| harness | candidate tier + production kernels vs full-scalar f32-KV oracle (`tge fidelity`) |

Corpus build:

```powershell
$files = @("docs\architecture.md","docs\report-0.1.md","docs\report-0.2-verification.md",
           "docs\benchmarks.md","docs\roadmap.md","docs\hardware-support.md",
           "docs\fused-correction.md","README.md")
$all = ($files | ForEach-Object { (Get-Content $_ -Raw) + "`n`n" }) -join ""
Set-Content -Path "$env:TEMP\kilo\corpus_natural_8k.txt" -Value $all -Encoding ASCII
```

Unlike `architecture.md` x N, whose first 8,199 tokens contain three copies of the same file, no scored position here has seen its own text before.

## 2. Results

### Natural-text 8k ladder (8,199 positions)

| mode | kept | mean KL | worst KL | max |diff| | top-1 | cost vs oracle |
|---|---:|---:|---:|---:|---:|---:|
| f16, dense | 8,199 | **2.312e-06** | 1.054e-03 | 8.051e-02 | **8191/8199 (99.90%)** | **−0.10pp** |
| i8, dense | 8,199 | 3.054e-02 | 5.850e+00 | 1.274e+01 | **7488/8199 (91.33%)** | **−8.57pp** |
| sparse 512/2 | 4,357 (53%) | 1.376e-01 | 1.160e+01 | 1.797e+01 | **7096/8199 (86.55%)** | **−13.35pp** |
| sparse 512/4 | 2,436 (30%) | 2.573e-01 | 1.176e+01 | 1.877e+01 | **6626/8199 (80.81%)** | **−19.09pp** |

All four arms are deterministic per binary; runs took 522-625 s each.

### Corpus effect at 8,199 positions (repeating vs distinct documents)

| tier | `architecture.md` x9 (repeat) | distinct docs (natural) | effect of repetition |
|---|---:|---:|---|
| f16 | 8197/8199 (99.98%) | 8191/8199 (99.90%) | −0.08pp (negligible) |
| i8 | 7920/8199 (96.60%) | 7488/8199 (91.33%) | **+5.27pp flattering** |
| sparse 512/4 | 5533/8199 (67.49%) | 6626/8199 (80.81%) | **−13.32pp penalising** |
| sparse 512/16 | 4356/8199 (53.13%) | not re-run | — |

The repetition effect is tier-specific: a repeated document makes i8 look better (the model's repeated-text predictions are easier, so quantization flips fewer argmaxes) and makes sparse look worse (the removed middle rows are the repeated copies the model is relying on, so structural omission flips more). Neither direction generalises, which is why both corpora must be reported.

### Natural-text context behaviour (first-copy slices, no repetition within the scored range)

| tier | 2k positions | 8k positions | delta |
|---|---:|---:|---:|
| f16 | 1998/1999 (99.95%) | 8191/8199 (99.90%) | −0.05pp (holds) |
| i8 | 1800/1999 (90.05%) | 7488/8199 (91.33%) | **+1.28pp (flat)** |
| sparse 512/4 | 1776/1999 (88.85%) | 6626/8199 (80.81%) | **−8.04pp (compounds)** |

The 2k rows are the first-copy slices measured in `idea-invented-sparse-geometry-sweep-measured.md` (sparse) and `idea-invented-i8-corpus-dependence-correction.md` (i8); at 2k each slice sits inside the first document, so the comparison is repetition-free on both sides.

## 3. Reading

- **f16 is the only quality-preserving 8k mode on realistic text**: 99.90%, −0.10pp, KL 2.3e-6. Eight flips in 8,199 positions, all near-ties.
- **i8 costs ~9pp on natural text at both 2k and 8k.** It is flat with context, not compounding, and it is nowhere near its 98.6% bar. It remains a capacity mode (KV quartered) and its continuous-batching admission win stands, but "i8 holds quality at 2k" must be quoted as 90.05% on prose, not 99.9% on repeated text. The previous report's "i8 error compounds" line is superseded: what compounded was the repeating corpus's difficulty falling with length.
- **Sparse compounds on natural text**: 11.15pp at 2k -> 19.09pp at 8k. Even the mildest shipped geometry (stride 2, 53% kept) costs 13.35pp at 8k. i8 beats every sparse geometry on quality at 8k while also quartering KV — the only thing sparse still wins is decode speed (1.36-1.77x, `idea-invented-sparse-8k-reopened.md`) and it pays 13-19pp for that on real text.
- **Decision matrix stands, with corrected magnitudes**: quality-first f16 dense (free); speed-first `--speculative --sparse` with 13-19pp stated; capacity mode i8 with ~9pp stated. Speculation remains quality-neutral and workload-quantified.
- **Method rule for the library**: never price a KV/attention tier on a self-repeating corpus. Repetition moves tier costs by up to 13pp in tier-specific directions. The natural-document corpus (eight distinct docs, 108,826 B) is the reference recipe from now on.

## 4. Falsifier (stated before the runs)

**CONFIRM the matrix ordering** if f16 stays within 0.5pp of the oracle, i8 lands below its 98.6% bar, and sparse lands below i8 on the same corpus. **REVISE** if any ordering flips (e.g. sparse above i8). Measured f16 −0.10pp, i8 −8.57pp, sparse/2 −13.35pp, sparse/4 −19.09pp: ordering f16 > i8 > sparse/2 > sparse/4 confirmed on natural text.

## 5. File anchors (current HEAD, read-before-write verified)

- `apps/cli/main.cpp:2520-2545` — fidelity reference (f32 KV, scalar kernels, Scalar attention) vs candidate tier; makes the KL a tier price.
- `apps/cli/main.cpp:2699+` — fidelity parser (`--kv`, `--sparse`, `--sparse-window`, `--sparse-stride`).
- `src/models/qwen2.cpp:117-134` — `store_kv_q8` per-32 quantization at store time.
- `src/models/qwen2.cpp:1329-1362` — sparse dispatch and `uses_sparse`.
- `src/cpu/avx2/attention_avx2.cpp:263,367` — strided sparse kernels (f32/f16) whose kept-set the sparse arms price.
- `reports/idea-invented-i8-corpus-dependence-correction.md` — the corpus-dependence diagnosis and the 2k natural i8 row.
- `reports/idea-invented-kv-tier-quality-8k-measured.md`, `idea-invented-sparse-8k-quality-correction.md` — the repetition-corpus 8k rows this cycle re-prices.
- `reports/idea-invented-sparse-geometry-sweep-measured.md` — the 2k natural sparse row (88.85%).
- `reports/idea-invented-8k-decision-matrix-closeout.md` — the matrix whose magnitudes are corrected here.

## 6. Reimplementation recipe

1. Build the natural corpus (see §1). Confirm it is long enough with a short-context run; the harness prints `corpus : N positions`:
   ```text
   tge fidelity --model qwen2.5-0.5b-instruct-q8_0.gguf --corpus <corpus> --context 64 --tokens 10 --kv f16 --threads 10 --gpu off
   ```
2. Run each arm (each ~9-10.5 min):
   ```text
   tge fidelity --model qwen2.5-0.5b-instruct-q8_0.gguf --corpus <corpus> \
     --context 8704 --tokens 8200 --kv f16 --threads 10 --gpu off \
     [--sparse --sparse-window 512 --sparse-stride 2|4]
   ```
   and the same with `--kv i8` and no sparse. Expect f16 8191/8199 (KL 2.312e-06), i8 7488/8199 (3.054e-02), sparse/2 7096/8199 (1.376e-01), sparse/4 6626/8199 (2.573e-01).
3. Do not substitute a self-repeating corpus; the tier costs move by up to 13pp and the ordering's magnitudes change.
4. No code ships.

## 7. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56, cycle gate)
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed

corpus: 8 distinct docs -> 108,826 B; hard-link scan shows no repeated document in the 8,199-token slice
fidelity (0.5B Q8_0, ctx 8704, 8200 tokens, 10 threads, gpu off):
  --kv f16                          -> KL 2.312e-06, worst 1.054e-03, max|diff| 8.051e-02, 8191/8199, 573 s
  --kv i8                           -> KL 3.054e-02, worst 5.850e+00, max|diff| 1.274e+01, 7488/8199, 522 s
  --kv f16 --sparse --stride 2      -> KL 1.376e-01, worst 1.160e+01, max|diff| 1.797e+01, 7096/8199, 625 s
  --kv f16 --sparse --stride 4      -> KL 2.573e-01, worst 1.176e+01, max|diff| 1.877e+01, 6626/8199, 605 s
git status --short                                -> clean (no code touched)
python aikit.py check aegean-tge                  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
