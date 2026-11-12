# MEASURED: single-request prefill parity under the path-invariant change — identical ids and neutral timing (1-token prompt now also kq8)

- **Question**: `idea-invented-continuous-kq8-path-invariant-keep.md` changed `batch_matvec` from `tokens > 1` to `tokens >= 1`, which also affects a **single-request** `generate`: a 1-token prompt chunk now takes the kq8 kernel instead of the float batch kernel. That is outside continuous batching, and the previous cycle verified only the batch path. This cycle checks the non-continuous surface: single-prompt prefill ids/timing, and a full 2,000-token prefill, on the path-invariant build vs the shipped binary.
- **Cycle date**: 2026-11-11
- **Verdict**: **MEASURED — no regression.** On `qwen2.5-0.5b-instruct-q4_k_m`, `--threads 10 --gpu off`:
  - **20-token prompt, n=32**: shipped `prefill 20 tokens in 130.6 ms (153.1 tok/s)`, path-invariant `123.6 ms (161.8 tok/s)`; generated ids **identical** (`C0-06-DA…`).
  - **2,000-token prompt** (`--context 2560`, n=4): shipped `7353.3 ms (272.0 tok/s)`, path-invariant `7239.7 ms (276.3 tok/s)`; ids **identical** (`576 3974 13876 38835`).
  - No output change on the generate path despite the 1-token chunk now using kq8: the kq8 batch kernel at `tokens == 1` is bit-identical to the float batch kernel for the same row on these shapes (and the 2,000-token prefill was already kq8 for every chunk > 1; only the final 1-token remainder changes path, and it stayed exact).
- **Tree state at close**: spike reverted from `src/models/qwen2.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Setup

- Shipped binary: `build/idea/tge-cont-float-base.exe` (HEAD, `float_only = true`, `tokens > 1`).
- Path-invariant spike: HEAD with the eight `float_only` arguments at `false` and the condition at `tokens >= 1`.
- Harnesses:
  - 20-token prose prompt, `--n 32 --print-ids` (decode after a short prefill).
  - 2,000-token periodic prompt (`fox x200`), `--context 2560 --n 4 --print-ids` (prefill dominated).
- Compare prefill t/s and full generated-id strings.

## 2. Results

| workload | shipped | path-invariant | ids |
|---|---|---|---|
| 20-token prompt, n=32 | 130.6 ms / 153.1 tok/s | 123.6 ms / 161.8 tok/s (1.06x) | identical |
| 2,000-token prompt, n=4 | 7353.3 ms / 272.0 tok/s | 7239.7 ms / 276.3 tok/s (1.02x) | identical |

Both timings are same-window single runs; the deltas are inside this box's noise and not claimed as wins. The claim is parity: no slowdown and byte-identical output on the non-continuous path.

## 3. Reading

- The 1-token chunk path change is safe: the kq8 batch kernel's per-row quantization and dots do not depend on token count, and at `tokens == 1` it equals the float batch row on these shapes (ids identical on both workloads).
- This closes the last untested surface of the path-invariant change: batch occupancy (3/3 audit shapes), continuous throughput (12 pairs), generate thinking set (5/5), and now single-request short and long prefill (2/2 ids identical).
- Promotion to default remains human-gated; if promoted, the invariant to keep is "path is chosen by kernel availability, never by occupancy or prompt length", and the solo-vs-busy audit plus these two parity checks are the regression set.

## 4. Reimplementation recipe (re-run the parity check)

1. Build shipped and path-invariant binaries as in `reports/idea-invented-continuous-kq8-path-invariant-keep.md` §8 (eight `float_only` flips + `tokens >= 1`).
2. Run `generate --model <q4_k model> --prompt <20-token prose> --n 32 --threads 10 --gpu off --print-ids` on both; compare ids and prefill t/s.
3. Run `generate ... --prompt <fox x200> --context 2560 --n 4 --threads 10 --gpu off --print-ids` on both; compare ids and prefill t/s.
4. Expected: ids identical in both, timing within noise. Any id difference kills the path-invariant change for the generate path and must be investigated before defaulting.

## 5. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge generate ... --prompt "<20 tok>" --n 32 --threads 10 --gpu off --print-ids
  shipped 130.6 ms (153.1 t/s) / path-invariant 123.6 ms (161.8 t/s), ids identical
tge generate ... --prompt "<fox x200>" --context 2560 --n 4 --threads 10 --gpu off --print-ids
  shipped 7353.3 ms (272.0 t/s) / path-invariant 7239.7 ms (276.3 t/s), ids identical
git checkout -- src/models/qwen2.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. Shipped baseline lives at `build/idea/tge-cont-float-base.exe`; the committed tree is unchanged except this report.
