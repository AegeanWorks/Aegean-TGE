# Benchmarks

**Canonical numbers: [measurement-log.md](measurement-log.md).** Every measured
value, every change with its before and after, and every rejected experiment
lives there and is updated with each change.

This file does not repeat them. It explains how a measurement is taken here, what
each metric actually means, and the four traps this project has already fallen
into — so the next person measures the right thing the first time.

## How a measurement is taken

```powershell
.\scripts\measure.ps1 > before.txt
# ... change ...
.\scripts\measure.ps1 > after.txt
Compare-Object (Get-Content before.txt) (Get-Content after.txt)
```

The snapshot prints the selected kernels plus one line per benchmark, so the diff
shows exactly what moved. Within a run each benchmark reports its best of N
repetitions; across runs, compare most lines and only believe deltas above about
20 percent unless both sides were measured back to back.

Every number in the log states the model file, thread count and shape it came
from. A figure without those is not a measurement.

## Why the numbers move

This machine holds 2-4 GB free of 16, with VS Code, Defender and Memory
Compression taking the rest. Two runs of `llama-bench` on the same file and
binary differed by 29 percent in one session; TGE's own prefill spans 170-201
t/s.

The practical rules that follow:

- Ratios measured back to back in one session are trustworthy; absolute values
  compared across sessions are not.
- A single sample is not evidence. The one time a headline was quoted from one
  run, it was 29 percent wrong.
- Prefer best-of-N for throughput and median for latency. Say which.

## What each metric means

| Metric | What it bounds | Scales with |
|---|---|---|
| `bench gemm` | prompt processing arithmetic | matrix size; compute-bound |
| `bench gemv` | the decode primitive, one weight stream | threads; bandwidth-bound |
| `bench mem` | the ceiling everything memory-bound sits under | threads |
| decode t/s | what a user feels; bytes per weight over bandwidth | quantisation, thread count |
| prefill t/s | prompt cost | GEMM throughput |

Decode and prefill are different problems and are never comparable to each other.
Decode reads every weight once per token and is bounded by bandwidth; prefill
reuses each weight across a token group and is bounded by arithmetic. A change
that helps one routinely does nothing for the other.

## The reference point

llama.cpp is run on the same file, the same machine and the same thread count.
It serves two purposes: a throughput comparison, and a correctness oracle.
Greedy output is diffed token for token, including across prefill chunk
boundaries, but parity is only a first check: near-ties are common, so a
vectorised or reordered kernel can diverge after a few dozen tokens without
being less faithful. `tge fidelity` prices that difference as KL against the
scalar reference; the current llama comparison and the near-tie that flips are
in measurement-log section 19.

## Four traps this project has already fallen into

1. **Instruction counting instead of port analysis.** Five optimisations looked
   cheaper by instruction count and measured worse. The clearest case was the
   integer inner loop: fewer instructions per multiply, but `psignb` contends for
   the single port that issues it. Counting ports would have predicted the loss.

2. **Benchmarking only the favourable shapes.** The VNNI kernel was 13 percent
   faster on the output projection and 2.4x slower on a 896x896 projection. The
   first benchmark covered only the large matrices and reported a win that did
   not generalise. Measure every shape the model actually uses.

3. **Single-sample comparisons.** Covered above; it produced a 67 percent figure
   that was really 13-33 percent.

4. **Attributing a whole-system number to one change.** "18.2 to 246.5 GFLOPS"
   bundled three code changes and a 1-to-16-thread comparison. The ablation took
   an afternoon and showed the pool was most of it. When a number moves, ablate
   before concluding.

## Reproducing

```powershell
. .\scripts\env.ps1
cmake --preset release
cmake --build --preset release

.\build\release\bin\tge.exe bench gemm --m 2048 --n 2048 --k 2048 --repeat 3
.\build\release\bin\tge.exe bench gemm --m 2048 --n 2048 --k 2048 --kernel unpacked
.\build\release\bin\tge.exe bench gemv --rows 4096 --cols 4096 --repeat 5
.\build\release\bin\tge.exe bench batch --rows 4864 --cols 896 --tokens 64 --repeat 3
.\build\release\bin\tge.exe bench batch --rows 896 --cols 4864 --tokens 1 --repeat 5 --format q6k
.\build\release\bin\tge.exe bench mem --size-mb 256 --repeat 5
```

`bench batch --format` takes `q8|q4k|q5k|q6k|q50|q51` (row bytes follow the
format's block: 34 for Q8_0, 144 per 256 cols for Q4_K, 176 for Q5_K, 210
for Q6_K, 22 per 32 cols for Q5_0, 24 per 32 cols for Q5_1). K-quant inputs are random bytes, so the rate measures unpack +
dot throughput, not any particular model.

The llama.cpp comparison needs its own build; see the log for the exact commit
and flags used, since a different build is not a comparison.
