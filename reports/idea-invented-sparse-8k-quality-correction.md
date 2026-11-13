# CORRECTION: sparse quality at 8k positions — the 2k proxy understated it 6x (stride 16 costs 14.4pp, not 2.25pp; even stride 2 leaves top-1 at 75.8%)

- **Idea**: The last three cycles priced `--sparse` at 8k for speed (`idea-invented-sparse-8k-reopened.md`: 1.36-1.77x decode; `idea-invented-sparse-prefill-8k-measured.md`: 1.06-1.15x prefill; `idea-invented-code-8k-spec-sparse-measured.md`: 1.61-1.83x combined) and the geometry sweep (`idea-invented-sparse-geometry-sweep-measured.md`) tuned the kept budget using a **2k-position fidelity proxy**, concluding that `--sparse-stride 16` costs 2.25pp top-1 at 8k. The open cell is the actual quality at 8k positions: the tier's loss compounds through the KV state, so the 2k proxy may not transfer. Run the fidelity harness at 8,199 positions for dense, stride 2, stride 4 (default), and stride 16.
- **Cycle date**: 2026-11-13
- **Verdict**: **CORRECTED — sparse at 8k is a heavily lossy mode, and the geometry sweep's stride-16 guidance is withdrawn.** Qwen2.5-0.5B-Instruct-Q8_0, `architecture.md` x 9 corpus, `--context 8704 --tokens 8200 --kv f16 --threads 10 --gpu off`, 8,199 teacher-forced positions:
  - **dense**: KL **1.145e-06**, worst 5.606e-04, top-1 **8197/8199 (99.98%)**.
  - **sparse 512/2** (~4,357 kept, 53%): KL 1.250e+00, top-1 **6216/8199 (75.82%)** — **−24.2pp**.
  - **sparse 512/4 (default)** (~2,436 kept, 30%): KL 1.630e+00, top-1 **5533/8199 (67.49%)** — **−32.5pp**.
  - **sparse 512/16** (~996 kept, 12%): KL 2.382e+00, top-1 **4356/8199 (53.13%)** — **−46.9pp**.
  - **The 2k proxy was 6.4x too kind on stride 16**: 2.25pp measured at ctx 2048 vs **14.37pp** at ctx 8199 for the same geometry. The 2k corpus also understates the default tier itself: 88.85% there vs 67.49% at 8k.
  - **Speed is unchanged** (prior cycles): default 8k decode 1.36-1.77x over dense and prefill 1.06-1.15x; stride 16 adds ~1.28x decode over default. Those numbers stand; only the quality framing was wrong.
- **Tree state at close**: no code touched (shipped flags only); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Qwen2.5-0.5B-Instruct-Q8_0 |
| corpus | `docs/architecture.md` repeated 9x (107,624 bytes) |
| positions | 8,199 usable, `--context 8704 --tokens 8200` |
| KV / threads | f16 / 10 |
| harness | full-scalar oracle vs AVX2-FMA production (`tge fidelity`) |
| arms | dense, `--sparse` with (window 512, stride 2), (512, 4), (512, 16) |

Corpus construction:

```powershell
$src = Get-Content projects\aegean-tge\docs\architecture.md -Raw
$corpus = ($src + "`n") * 9
Set-Content -Path "$env:TEMP\kilo\corpus_8k.txt" -Value $corpus -Encoding ASCII
```

The corpus is repetitive by design (the 2k proxy used the same file x2), so the 8k difference is context length, not content novelty.

## 2. Results

| arm | kept @8199 | kept % | mean KL | worst KL | max abs diff | top-1 | vs dense | time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 8,199 | 100% | **1.145e-06** | 5.606e-04 | 9.829e-02 | **8197/8199 (99.98%)** | — | 507 s |
| sparse 512/2 | 4,357 | 53% | 1.250e+00 | 1.940e+01 | 2.710e+01 | 6216 (75.82%) | −24.16pp | 549 s |
| sparse 512/4 (default) | 2,436 | 30% | 1.630e+00 | 1.833e+01 | 2.858e+01 | 5533 (67.49%) | −32.49pp | 568 s |
| sparse 512/16 | 996 | 12% | 2.382e+00 | 1.825e+01 | 3.005e+01 | 4356 (53.13%) | −46.85pp | 668 s |

### The 2k proxy vs actual 8k (same corpus family, same model)

| geometry | 2k positions (ctx 2048) | 8k positions (ctx 8704) | error of the proxy |
|---|---:|---:|---:|
| 512/4 default | 1776/1999 (88.85%) | 5533/8199 (67.49%) | 21.4pp |
| 512/16 | 1731/1999 (86.59%) | 4356/8199 (53.13%) | 33.5pp |

The proxy's *deltas* were wrong too: at 2k, stride 16 cost 2.25pp vs default; at 8k it costs 14.37pp. Dilution that is nearly free at 2k removes a third of the model's context at 8k.

## 3. Reading

- **Sparse quality compounds with context.** At 8k even the default tier flips one top-1 token in three. The mechanism is the KV state: each layer attends over the sparse subset, so early omissions feed later positions through the cache; a subset that is 30% of an 8k span is not equivalent to 30% of a 2k span, and the harness measures the accumulated result.
- **No tested geometry is a quality-preserving 8k mode.** Stride 2 keeps 53% of the rows and still costs 24pp. If sparse is used at 8k, it is an explicitly lossy speed mode (1.4-1.8x decode, 1.1-1.2x prefill for 24-47pp top-1); it is not a general long-context default, and it should not be quoted as "the robust 8k lever" without that price.
- **The geometry-recipe guidance is withdrawn**: do not use `--sparse-stride 16` as an 8k speed option on quality-sensitive work. The keep-grade budget mode remains technically valid as a speed knob, but this cycle prices its 8k quality cost as unacceptable outside deliberately lossy workloads.
- **Speculation composition is unaffected numerically but must be read correctly**: `--speculative --sparse` accepts 5.33/step against the sparse model's own logits; that is not evidence of fidelity. The speed stack (1.71-1.83x on code) remains true as a speed claim.
- **What a real long-context mode needs**: either a much larger kept budget (quality at 53% kept is still 75.8%, so the curve is steep — extrapolating, ~90% may need >80% kept, i.e. dense), a scoring/selection method that keeps the rows the model actually uses (IDEA-1451's scored middle was killed on speed at 2k; at 8k its quality case is stronger but its speed cost is fatal), or a KV-tier/quantization lever instead of dropping rows. The honest long-context recommendation today is dense or f16/i8 KV tiers; sparse is opt-in for lossy workloads only.

## 4. Falsifier (stated for this diagnostic)

**KEEP the 8k sparse guidance** if sparse top-1 at 8,199 positions is within 2pp of dense for some tested geometry. **CORRECT/WITHDRAW** otherwise. Measured −24.16pp at the mildest geometry (stride 2): withdraw. The dense control's 99.98% rules out harness drift or corpus difficulty as the cause.

## 5. File anchors (current HEAD, read-before-write verified)

- `include/tge/models/qwen2.hpp` — `SparseConfig{sink=4, window=512, stride=4}`.
- `src/models/qwen2.cpp:222-260` — `build_sparse_index` geometry; `:1329-1342`, `:1357-1361` — sparse dispatch and `uses_sparse`.
- `src/cpu/avx2/attention_avx2.cpp:263,367` — strided kernels; their kept-set walk is what the fidelity harness prices.
- `apps/cli/main.cpp:2699+` — fidelity parser (`--sparse`, `--sparse-window`, `--sparse-stride`).
- `reports/idea-invented-sparse-geometry-sweep-measured.md` — the 2k-proxy report this cycle corrects.
- `reports/idea-invented-sparse-8k-reopened.md`, `idea-invented-sparse-prefill-8k-measured.md`, `idea-invented-code-8k-spec-sparse-measured.md` — the 8k speed results (unchanged).
- measurement-log §50 — the ctx-1024 sparse fidelity reference (KL 2.6e-2, 975/1023).

## 6. Reimplementation recipe

1. Build the corpus (see §1); confirm the position count with a short-context run:

   ```text
   tge fidelity --model qwen2.5-0.5b-instruct-q8_0.gguf --corpus <corpus8k> \
     --context 64 --tokens 10 --kv f16 --threads 10 --gpu off   -> corpus: 9 positions
   ```

2. Run each arm (each costs ~8-11 minutes; the harness runs scalar and AVX2 in lockstep):

   ```text
   tge fidelity --model qwen2.5-0.5b-instruct-q8_0.gguf --corpus <corpus8k> \
     --context 8704 --tokens 8200 --kv f16 --threads 10 --gpu off \
     [--sparse --sparse-window 512 --sparse-stride S]
   ```

   Expect dense KL 1.145e-06 / 8197/8199; stride 2 1.250 / 6216; stride 4 1.630 / 5533; stride 16 2.382 / 4356.
3. Do not use the 2k fidelity proxy to price long-context sparse geometry; run at the target context.
4. No code ships.

## 7. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56, cycle gate)
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed

corpus built: architecture.md x 9 -> 107,624 bytes; corpus line confirms 8,199 positions at ctx 8704

fidelity (0.5B Q8_0, ctx 8704, 8200 tokens, f16, 10 threads):
  dense        -> KL 1.145e-06, worst 5.606e-04, max|diff| 9.829e-02, 8197/8199, 507 s
  sparse 512/2 -> KL 1.250e+00, worst 1.940e+01, max|diff| 2.710e+01, 6216/8199, 549 s
  sparse 512/4 -> KL 1.630e+00, worst 1.833e+01, max|diff| 2.858e+01, 5533/8199, 568 s
  sparse 512/16-> KL 2.382e+00, worst 1.825e+01, max|diff| 3.005e+01, 4356/8199, 668 s
git status --short                                -> clean (no code touched)
python aikit.py check aegean-tge                  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
