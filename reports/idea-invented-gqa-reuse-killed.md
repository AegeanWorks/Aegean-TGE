# INVENTED: Token-outer GQA reuse for i8 attention — KILLED (0.81–0.96x at 8k, 0.83–0.94x at 4k)

- **Idea**: Invert the loops in `attention_group_avx2` for the int8 KV tier: process one KV token at a time and reuse the loaded/widened K and V rows across all query heads in the GQA group, keeping the same per-head online softmax. The head-outer kernel currently reads each row once per head, so a 3-head group reads K/V three times.
- **Cycle date**: 2026-11-07
- **Verdict**: **KILLED on the kernel falsifier.** The token-outer kernel is numerically equivalent (max normalized difference 5.9e-6 at 8k, 3.0e-6 at 4k) but consistently **slower**: 0.81–0.96x at 8192 keys and 0.83–0.94x at 4096 keys. No model wiring was attempted. The cycle-27 recommendation to attack the value-accumulation path is now closed at this loop layout.
- **Tree state**: spike reverted from `include/tge/cpu/kernels.hpp`, `src/cpu/avx2/attention_avx2.cpp`, `tests/test_attention.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; nothing pushed.

## 1. Mechanism

Decode at position ≥256 uses the GQA schedule: for each KV head and span chunk, `attention_group_avx2` receives `group_heads = 3` query heads and one K/V head. The current kernel loops **heads outer, tokens inner**, so the same 128-byte key row and 128-byte value row are loaded and widened three times (once per query head).

The spike implements `attention_group_q8_reuse_avx2` with **tokens outer, heads inner**:

1. One key-lane widen per 8 values feeds all three QK accumulators.
2. One value-lane widen per 8 values feeds all three value accumulators.
3. Per-head online softmax (`maximum`, `sum`, alpha rescale) is unchanged; per-head arithmetic order is preserved.
4. Non-q8, group > 8, or null buffers fall back to the existing kernel.

The 8192-key / 4096-key A/B is the stage-1 stand-in for the model's 8k decode, where each task sees a ~4096-token chunk.

## 2. Falsifier

- **PROCEED to model wiring** only if the token-outer kernel is **≥1.4x** the head-outer kernel, with equivalent results; the model gate would then be ≥1.15x decode at 8192.
- **KILL** if the kernel is neutral or slower.

Measured: slower at both spans; kill.

## 3. Measured numbers

Temporary `attention_group_q8_reuse_stage1` test: 3 query heads × 128 dims, random int8 K/V with per-32 scales, 20 iterations per round, best of 3 rounds.

### 8192-key span

| run | head-outer | token-outer | speedup |
|---:|---:|---:|---:|
| 1 | 1.648 ms | 1.723 ms | 0.96x |
| 2 | 1.571 ms | 1.915 ms | 0.82x |
| 3 | 1.482 ms | 1.824 ms | 0.81x |
| 4 | 1.655 ms | 1.729 ms | 0.96x |
| **range** | 1.48–1.65 ms | 1.72–1.92 ms | **0.81–0.96x** |

Max normalized difference between kernels: **5.879e-06**.

### 4096-key span (matches the model's chunk size)

| run | head-outer | token-outer | speedup |
|---:|---:|---:|---:|
| 1 | 0.945 ms | 1.132 ms | 0.83x |
| 2 | 0.865 ms | 0.916 ms | 0.94x |
| 3 | 0.753 ms | 0.890 ms | 0.85x |
| **range** | 0.75–0.95 ms | 0.89–1.13 ms | **0.83–0.94x** |

Max normalized difference: **2.960e-06**. The ordering is stable across spans and runs, so the result is not a single-window artifact.

## 4. Why it lost

- **The rows were already cache-resident.** One KV head's chunk is 4096 × 128 B = 512 KiB of keys plus 512 KiB of values, inside the shared L3 and largely the L2 of the working core. The head-outer pass re-reads from L2, not DRAM, so eliminating repeat reads saves little bandwidth.
- **Register pressure.** Token-outer needs six QK accumulators (2 per head) plus three value accumulators plus shared widened K/V registers. The head-outer kernel needs only two QK accumulators and one value accumulator at a time, so it streams without spills.
- **Softmax bookkeeping moved into the row loop.** Three exp/branch updates per token now sit between the row load and the value FMA; head-outer keeps each head's softmax branch on a hot, independent stream.
- The i8 value pass is not load-bound at this working-set size; it is FMA/accumulator-bound, so reuse cannot pay.

## 5. Disposition

- Keep the current head-outer `attention_group_avx2`.
- Do **not** wire the token-outer kernel; the value/softmax attention path is closed at this loop layout.
- A future attempt needs a different strategy, not a loop inversion: for example, a fused micro-kernel that keeps the whole group's accumulators in registers across a short token block, or a KV layout with the group heads' values interleaved so one load serves the FMAs without extra accumulator pressure.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 146/146 with the stage-1 bench; 145/145 after revert
# 8192 keys: head-outer 1.48-1.65 ms, token-outer 1.72-1.92 ms (0.81-0.96x)
# 4096 keys: head-outer 0.75-0.95 ms, token-outer 0.89-1.13 ms (0.83-0.94x)
# maxrel 5.879e-06 / 2.960e-06
git checkout -- include/tge/cpu/kernels.hpp src/cpu/avx2/attention_avx2.cpp tests/test_attention.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
