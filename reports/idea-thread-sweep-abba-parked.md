# Thread Sweep ABBA — STILL INCONCLUSIVE, PARKED until cool box

- **Idea**: Follow-up to `idea-thread-sweep-inconclusive.md`: re-test 4T vs 6T prefill on a stable box with order-balanced ABBA (4,6,6,4 then 6,4,4,6), 0.5B-Q8_0, 2048-tok prompt, n=8.
- **Cycle date**: 2026-11-07
- **Verdict**: INCONCLUSIVE — park. Adjacent-pair deltas split 2-2 (6T +2%/+13% in part 1; 4T +8%/+5% in part 2) with run-to-run noise of ±10-30% on both prefill and decode, dwarfing any thread-count effect. No default change.
- **Tree state at close**: clean, no diff, stash untouched. No code touched (measurement only). Nothing pushed.

## 1. Measured numbers (prefill t/s, then decode t/s)

Part 1 (order 4,6,6,4): 4T 261.1/37.9, 6T 266.2/47.1, 6T 250.7/46.6, 4T 221.4/35.6.
Part 2 (order 6,4,4,6): 6T 232.7/65.6, 4T 251.7/52.2, 4T 254.3/52.8, 6T 241.5/54.2.

Adjacent pairs: P1 6T +2%, P2 6T +13% (4T leg anomalously low), P3 4T +8%, P4 4T +5%. The earlier best-of-2 signal (4T +20%) does not reproduce under order balancing; absolute levels drift ±15% across the session independent of thread count.

## 2. Why parked, and what unparks it

The effect size (if any) is smaller than same-config run-to-run variance on this box, so no A/B on this box can resolve it either way. Unpark only on a cool, quiet box with interleaved best-of-5 per arm (ABBAABBA order), same shape; keep iff 4T wins ≥4/5 pairs by ≥10% with identical transcripts.

## 3. Exact commands run

- 8× `tge generate --model 0.5B-Q8_0 --tokens "<2048 ids>" --n 8 --threads {4|6} --gpu off`, order 4,6,6,4,6,4,4,6. No code changes; no fidelity owed (no code touched).

(End of file.)
