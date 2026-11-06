# INVENTED: Norm-Phase Split Diagnostic — MEASURED (flat thirds 40/34/26; detect() 6.7us × ~100 calls/token is the only cross-cutting lever, ceilings 5%/0.7% — at/under bar, fix deprioritized)

- **Idea**: Attribute the ~11-14% decode "norm" phase: the bucket (`phase_norm_ns_`) grabs three different kernels (rms_norm + apply_rope + silu_gated), and the per-call cost (~15-21 µs for kilobytes of traffic) smells like fixed overhead, not work. Split it with temp sub-phase counters + a direct microbench probe, name the lever with ceiling math.
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED diagnostic, fix DEPRIORITIZED. The split is flat (no ≥50% bucket — fusion hopes dead), and the only cross-cutting lever is `cpu::detect()` at ~6.7 µs/call × 97–113 calls/token: ~50% of the norm bucket, ceiling ≈5% on 0.5B (at the bar, not above) and ≈0.7% on Llama-3B (sub-bar). Memoizing detect is behavior-identical and safe, but the loop bar says don't bother — recorded here as a valid-but-sub-bar recipe (§7).
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). Spike branch `spike-normsplit` created (zero commits) and deleted; all three spike files reverted with empty diff verified. `master` ahead of `origin/master` by 97 after this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked, reverted)

Temp sub-phase counters (`phase_rms_ns_/rope/silu_` + getters + reset) accumulated with the same `dt` as the norm bucket in `rms_norm` (2 paths), `apply_rope` (4 return paths), `silu_gated` (1 path), plus a temp `spike-normprobe` driver: 200k× `cpu::detect()` timing + 2000× direct kernel timings (rms896, rope1024 plain+interleaved, silu4864). One `normsplit:` print line in the decode phases block. All reverted.

## 2. Predicted gain — diagnostic, landed as attribution + ceilings

No end-to-end gain claimed (instrument only). Expected: one bucket ≥50% (fusion target) or a quantified cross-cutting lever. Got the second: detect() ≈ 6.7 µs × 2910 calls (0.5B 30-tok run) ≈ 19.5 ms ≈ 50% of the 39.3 ms bucket ≈ 5.3% of the 371 ms wall; Llama-3B 3390 × 6.7 µs ≈ 22.7 ms ≈ 28% of its 79.9 ms bucket ≈ 0.7% of wall.

## 3. Falsifier outcome — flat shares honestly reported, lever quantified

Stated: name a ≥50% bucket or report honestly + quantify the next lever. Shares are 40/34/26 (0.5B) and 43/30/27 (Llama-3B) — no fusion target exists (a norm/rope/silu fusion would need all three anyway and saves no traffic, only ~15-21 µs/call of mostly-fixed cost). The cross-cutting lever (detect) is quantified with ceilings (§2, §6) that sit at/under the 5% bar → fix deprioritized, not killed-on-principle (it remains valid-but-sub-bar; trigger: any future detect() cost growth or a narrower bar).

## 4. File anchors (master = pre-spike; spike reverted)

- `src/models/qwen2.cpp:1075-1133` (`rms_norm`/`apply_rope`/`silu_gated` — the norm bucket's three tenants; instrumented, reverted).
- `include/tge/models/qwen2.hpp` (temp counters/getters — reverted).
- `apps/cli/main.cpp` (temp `spike-normprobe` + `normsplit:` line — reverted).
- `src/core/cpu.cpp:230-309` (`detect()`: ~10 CPUID leaves + brand-string heap work per call — the cost being timed).
- `src/cpu/avx2/norm_rope_avx2.cpp:59-158` (clean kernels, no pool/detect inside — the fixed cost lives in the callers).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-normsplit` (zero commits, deleted at close). Baseline: build 54/54, `tge_tests` 141/141.
2. ANALYZE: bucket composition confirmed; silu-exp hypothesis checked against kernel bodies (vectorized exp ships; per-call times flat across 7 KB vs 58 KB of work → fixed-cost smell, not compute).
3. Instrumented (§1). Rebuilt 54/54, `tge_tests` 141/141.
4. Split runs: 0.5B story 10T (§6) + Llama-3B story 10T (§6); microbench probe ×2 (§6).
5. `aikit.py check aegean-tge` → 2/4 (test + secrets PASS; hygiene + guidelines fail pre-existing, same pair as every prior cycle).
6. Reverted all three files (`git checkout --`, empty diff), `git checkout master`, branch deleted, rebuilt master 54/54, `tge_tests` 141/141, `git status` clean, stash untouched.

## 6. Measured numbers

0.5B-Q8_0 story 31 tok, 10T, warm page cache: decode 30 tok 371.00 ms (80.86 t/s); decode phases matvec 317.9 (88%), attn 5.3 (1%), norm **39.3 (11%)**; normsplit **rms 15.8 (40%), rope 13.3 (34%), silu 10.2 (26%)** = 39.3 bucket exactly.

Microbench probe ×2: detect **6746/6663 ns/call**; rms896 0.18 µs; rope1024 4.67/4.91; rope_int1024 15.37/16.23; silu4864 1.85/2.01.

Llama-3B-Q4_K_M story 31 tok, 10T: decode 30 tok 3193.65 ms (9.39 t/s); decode phases 3076.8 (97%), 22.9 (1%), norm **79.9 (3%)**; normsplit **rms 34.4 (43%), rope 24.2 (30%), silu 21.3 (27%)**.

Per-call (calls = tokens × norms/tokens): 0.5B rms 10.7 µs (1470), rope 18.5 µs, silu 14.2 µs (720 each); 3B rms 20.1 µs (1710), rope 28.8 µs, silu 25.4 µs (840 each). Microbench-vs-insitu gap (rms kernel 0.18 µs in-probe vs ~4-13 µs unexplained in-situ beyond detect) = cache-cold data + 10-thread contention + timer; noted as unattributed remainder, not a lever (no fix direction).

No fidelity run owed (timers + reads only; kernels ran on temp buffers, model untouched). No thinking-set diff owed (same model behavior; only phase prints added).

## 7. Full reimplementation recipe (valid-but-sub-bar — do NOT prioritize)

1. `src/core/cpu.cpp` + `include/tge/core/cpu.hpp`: memoized accessor, e.g. `const CpuInfo& cached_cpu_info() { static const CpuInfo info = detect(); return info; }` (process-lifetime hardware/XCR0 props — all current uses are `.features.*` reads, none depends on re-detection).
2. Replace every `cpu::detect().features` call site (rms/rope/silu per-call ×97-113/token are the prize; batch_matvec/forward/dequant sites amortize but take the same accessor for uniformity).
3. Gate the port: build 54/54, `tge_tests` 141/141, story 1T/10T best-of-3 + interleave on 0.5B-Q8_0 (expect ≈+4-5% 10T decode, ≈0% prefill-shape change), one Llama-3B pair (≈+0.7%), fidelity digit-identity (same features → same paths), story/creative ids identical, `aikit.py check`.
4. Ship ONLY if a re-measure clears the bar in the port window (this cycle ceilings it at the bar, not above). Revisit trigger: detect() gains cost (more leaves/strings) or the bar narrows.

## 8. Verdict and why

**MEASURED: the norm phase is fully attributed and the only lever is priced out.** Flat thirds kill fusion hopes (no dominant bucket, and fusing saves no traffic — the per-call cost is mostly fixed). The detect() finding (≈5% 0.5B / 0.7% 3B ceiling) is real but sub-bar, so the loop banks the attribution and moves on. Next-cycle candidates: the 1517 epic's stage-2 skip harness remains the only live structural front; the human-gated port shelf grows (mprefix-table, sparse no-gather, batchsnap kind tags, timers fix, detect memoize).

## 9. Exact commands run and their results

- `git checkout -b spike-normsplit` → branch (zero commits, deleted at close).
- Baseline: `cmake --build --preset release` → 54/54; `tge_tests.exe` → 141/141.
- Instrument (§1); rebuild → 54/54; `tge_tests` → 141/141.
- `generate --model <0.5B-Q8_0> --threads 10 ... --n 31` → §6 split row; `spike-normprobe` ×2 → §6 probe rows; `generate --model <Llama-3B-Q4_K_M> ...` → §6 3B row.
- `aikit.py check` → 2/4 (test + secrets pass; pre-existing hygiene + guidelines fails).
- `git checkout -- <3 files>` → clean; `git checkout master`; `git branch -D spike-normsplit`; rebuild 54/54; tests 141/141; scratch none (no files written); report committed; no push.

(End of file.)
