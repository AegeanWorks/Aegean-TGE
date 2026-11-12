# MEASURED: thinking-set divergence of the per-32 kq8d max tier — 4/5 greedy prompts byte-identical, 1/5 diverges at token 40; sampled ids differ by design

- **Question**: The max tier (`--kq8d head,ffn,down,attn`) is priced at mean KL 4.382e-4 / 393/399 top-1 on the fidelity corpus. What does that price look like on the loop's fixed thinking set — which prompts keep their exact output, and when one flips, how far does it propagate? This is the thinking-set diff the loop owes for any numerics change; the tier had only fidelity numbers.
- **Cycle date**: 2026-11-11
- **Verdict**: **MEASURED.** Greedy (n=64, threads 10, f32 KV, exact default vs max tier):
  - **identical (64/64 ids): story, code, CJK, abstract** — 4 of 5 prompts.
  - **creative diverges at output token 40/63** (`Write the opening line of a novel about a lighthouse keeper who finds a message in a bottle:`). Up to token 39 the ids are equal; then default `6266 4815 2028 3488 8720 288 701` vs tier `9456 2028 3488 7612 279 1296 2442` — the suffix partially realigns around `2028 3488` and then continues differently. That is a single near-tie flip cascading through a high-entropy creative continuation, exactly what a 4.4e-4 KL implies at one of five prompts.
  - **sampled (temp 0.8, seed 7, story prompt): ids differ** (`B3-47-2E…` vs `4B-D0-78…`). Expected: the tier changes the probability vector, so the same uniform stream lands on different cumulative bins; sampled runs are not expected to be byte-identical under any lossy tier.
  - Same-window decode on the creative prompt: 24.31 vs 17.08 t/s (**1.42x**), consistent with the tier's keep numbers.
- **Tree state at close**: tier re-applied for the measurement then reverted from `src/cpu/avx2/dequant_batch_kq8_avx2.cpp`, `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp`, `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Setup

- Model: `Llama-3.2-3B-Instruct-Q4_K_M.gguf`, `--threads 10 --gpu off --kv f32`, greedy (default temp) unless stated, `--n 64`, `--print-ids`.
- Arms: default (exact fused float) vs `--kq8d head,ffn,down,attn` (per-32 integer, the tier priced at 4.382e-4 / 393/399).
- Fixed prompt set (the loop's "thinking set", defined here so future cycles can reuse it):
  1. **story**: "Once upon a time, there was a small dragon who lived in a quiet cave near the village."
  2. **creative**: "Write the opening line of a novel about a lighthouse keeper who finds a message in a bottle:"
  3. **code**: "Write a C function that reverses a singly linked list:"
  4. **cjk**: "用中文写一段关于秋天的短文："
  5. **abstract**: "The relationship between memory and identity is"

## 2. Results

| prompt | default ids | tier ids | first diff | verdict |
|---|---|---|---|---|
| story | 64 | 64 | - | identical |
| creative | 64 | 64 | **40** | diverges |
| code | 64 | 64 | - | identical |
| cjk | 64 | 64 | - | identical |
| abstract | 64 | 64 | - | identical |

Divergent slice at positions 36..46 (0-based):

- default: `701 1866 8736 1584 6266 4815 2028 3488 8720 288 701`
- tier: `701 1866 8736 1584 9456 2028 3488 7612 279 1296 2442`

The two sequences share positions 36-39, differ at 40, re-converge on `2028 3488` (41-42 tier, 41-42 default), then separate. A single flip, not a wholesale distribution shift.

Sampled check (temp 0.8, seed 7, story prompt, n=64): default `B3-47-2E…`, tier `4B-D0-78…` — different, as expected when the probability vector changes under a fixed uniform stream.

## 3. Reading

- **Greedy users of the tier get the exact default output on 4/5 of these prompts**, including code (where determinism matters most) and CJK. The creative prompt is the worst case: a high-entropy continuation at the top-1 near-tie boundary, where the tier's priced 98.50% top-1 shows up as one flip at position 40 of 64.
- The suffix after a flip is *not* garbage and partially re-converges; calling the output "same distribution, different draw" is accurate, but the draw can differ in the middle of a run.
- **Sampled runs must not be expected to match**, with or without speculation: a lossy tier changes the cumulative distribution, and any fixed-seed comparison is meaningless for parity. The fidelity KL/top-1 pair is the parity metric.
- Together with the fidelity pricing, this is the practical contract for the tier: exact by default; opt-in lossy tier = 98.5% top-1, 4/5 thinking-set prompts byte-identical greedy, creative runs may flip mid-output, sampled runs differ.
- The thinking set is now defined and reusable; record it as story/creative/code/cjk/abstract with the exact prompt strings above.

## 4. Reimplementation recipe (re-run the measurement)

1. Re-apply the per-32 max tier (`reports/idea-invented-kq8d-full-stack-per32-keep.md` §8) and build + 171/171.
2. For each prompt in §1: `tge generate --model <gguf> --prompt "<prompt>" --n 64 --threads 10 --gpu off --print-ids [--kq8d head,ffn,down,attn]`.
3. Compare the `ids:` lines token-by-token; record the first differing index and the surrounding 10-token slice.
4. One sampled check: `--temp 0.8 --seed 7` on the story prompt; expect different ids (record hashes only).
5. Revert the tier. Expected: 4/5 greedy identical, creative first-diff ~40, sampled differs.

## 5. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge generate ... --n 64 --threads 10 --gpu off --print-ids [--kq8d head,ffn,down,attn]
  story/code/cjk/abstract -> 64/64 identical
  creative                -> first diff at 40; slices in §2
tge generate ... --n 64 --temp 0.8 --seed 7 ... story -> ids differ (B3-47-2E vs 4B-D0-78)
git checkout -- src/cpu/avx2/dequant_batch_kq8_avx2.cpp include/tge/models/qwen2.hpp \
  src/models/qwen2.cpp apps/cli/main.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The tier exists only in the rebuilt binary; the committed tree is unchanged except this report.
