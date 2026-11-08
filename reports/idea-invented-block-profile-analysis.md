# ANALYSIS: block prefill attention internal profile — value pass 68%, QK 30%, softmax 3%; no AVX2 headroom

- **Idea**: The report-only query-block prefill kernel halves 8k attention, but its remaining ~230 s at 8180 tokens was unattributed. Instrument the three passes (QK, softmax, value) to decide whether another iteration has headroom.
- **Cycle date**: 2026-11-08
- **Verdict**: **No further optimization recommended.** Thread-time at 8180 tokens: **value 67.8%**, QK **29.5%**, softmax **2.8%**. The value pass is memory-op bound (one accumulator load + one store per 8 values per key-token); the only competing reordering trades accumulator traffic for value-row re-reads at roughly 1:1, and faster arithmetic needs AVX-512 or lower-precision accumulation, both out of scope. QK's horizontal-sum overhead is inherent to the layout. Close the block-kernel optimization line; it remains an integration candidate at 1.6–2.3x attention.
- **Tree state**: spike reverted from `include/tge/cpu/kernels.hpp`, `src/cpu/avx2/attention_avx2.cpp`, `src/models/qwen2.cpp` and `apps/cli/main.cpp`; 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Mechanism

Re-applied the f32 query-block prefill kernel (cycles 33/36) with three `std::atomic` nanosecond accumulators, timed per head call:

- QK: key rows loaded once, dotted against every query in the block.
- Softmax: per-query max/exp/inverse over the causal prefix.
- Value: value rows loaded once, FMA'd into every query's accumulators.

`blockprof` is printed after the prefill phases. The counters sum across the prefill pool's threads; wall attention × thread count is the check on the effective parallelism.

## 2. Falsifier

- **ITERATE** if one pass is both dominant and structurally reducible.
- **CLOSE** if the dominant pass is at its memory/instruction limit.

Measured: value dominant and structurally limited.

## 3. Measured numbers — 8180-token prefill, f32 KV

| metric | value |
|---|---:|
| prefill wall | 436,105 ms (18.8 t/s) |
| prefill matvec | 190,513 ms |
| prefill attention (wall) | 229,911 ms |
| prefill norm | 13,975 ms |
| block thread-time: QK | 841,030 ms (**29.5%**) |
| block thread-time: softmax | 79,090 ms (**2.8%**) |
| block thread-time: value | 1,933,702 ms (**67.8%**) |
| block thread-time: total | 2,853,822 ms |

Total thread-time ÷ attention wall = **12.4 effective threads**, consistent with prefill's all-logical-processor pool rather than the 6 decode threads.

## 4. Why the value pass is the limit

Per (key, token), the value pass does, for each of 16 eight-value chunks:

```text
load V[8] (shared across the block's queries)
load accumulator[8]
FMA weight * V
store accumulator[8]
```

That is 512 B of accumulator read + 512 B write per key-token, versus 512 B of value data read once per key for all 64 tokens. The accumulator traffic dominates and is L1-resident but store-throughput-bound.

Alternatives and their trade:

- **Dim-chunk-outer with a register accumulator**: keeps one `__m256` across all keys, eliminating accumulator loads/stores, but re-reads each value row once per query (16 chunks × 64 tokens) — value traffic rises from one row per key to one row per key-token, offsetting most of the saving, and the score/weight array is re-read.
- **Lower-precision accumulators** (f16/bf16): halves the store width but accumulates thousands of terms at reduced precision; the quality risk is not acceptable for a prefill path whose fidelity is otherwise only accumulation-order noise.
- **AVX-512**: would provide 32 registers and 16-wide vectors, enabling a 2-token tile; the target CPU (i5-14400F) has AVX-512 disabled.
- **QK microkernel**: 4×4 or 8×8 blocking cannot fit the accumulators in 16 YMM registers; the horizontal sum per dot is inherent to the reduction layout, so QK is near its AVX2 limit too.

## 5. Disposition

- **Close the block prefill kernel optimization line.** The measured 67.8% value share is structural on this ISA; no further spike is justified.
- Keep the kernel as-is in the integration dossier: f32/f16 prefill, 1.6–2.3x attention, ids identical across all tested prompts.
- The remaining prefill levers, if any, are outside this kernel: matvec (34–44% of prefill) or algorithm-level changes (e.g., sparse attention), not more blocking.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
tge generate --model Llama-3B --tokens @ids8180.txt --n 8 --threads 6 \
  --gpu off --kv f32 --context 8192
# blockprof: qk 841.0 ms, softmax 79.1 ms, value 1933.7 ms
git checkout -- include/tge/cpu/kernels.hpp src/cpu/avx2/attention_avx2.cpp \
  src/models/qwen2.cpp apps/cli/main.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
