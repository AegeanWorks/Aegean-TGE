# VERIFICATION ROUND — table + S1 hold, hoist reverted (effect not robust)

- **Date**: 2026-11-07. Tree `e6410a7` (all three winner ports) re-measured against saved pre-port binaries on the live box.
- **Box state**: disturbed — identical-binary back-to-back matvec ±17%, `bench mem` read 53–62 GB/s across runs, background desktop load (browser, game launcher, remote desktop). Any ±20% effect is unresolvable in this state; ~70% effects resolve.

## 1. Multi-prefix table — RE-VERIFIED, stays

Temp driver re-added (reverted after): 0.5B-Q4_K_M 8×(120+32) single-CoW binary vs current tree: 2801.6 → 1661.0 ms (**1.69x**, vs 1.72x claimed), 2 POPULATE + 6 HIT, zero EVICT, ids byte-identical (`198 198 198 220 6565 406 220 82`). Effect dwarfs the noise. Keep-grade stands.

## 2. 1517-S1 batch populate — RE-VERIFIED, stays

Temp MATCH driver re-added (reverted after): `[cow-mp] POPULATE C=32 slot=0 (batch)` → `[cow-mp] HIT seq=1 C=32 slot=0` → **MATCH** (deterministic, box-independent). Overhead claim (never-negative) unchanged from landing data. Keep-grade stands.

## 3. Grouped-hoist — COULD NOT REPRODUCE, reverted (`2782b43`)

Four order-balanced interleaved pairs, 0.5B-Q8_0 `--kv i8` threads 10, current tree vs pre-hoist binary `tge-baseline.exe`, ids identical every run:

| span | baseline attn | current attn | delta |
|---|---|---|---|
| 3.9k | 92.2 | 100.0 | +8.5% |
| 3.9k | 88.4 | 104.4 | +18.0% |
| 3.9k | 85.8 | 91.9 | +7.1% |
| 7.0k | 159.0 | 176.2 | +10.8% |

The decode kernel file is byte-identical to the landed port (`git diff cb2a330..HEAD -- src/cpu/` empty), so the code under test IS the winner code — the effect flipped sign with machine state. The landing best-of-3 (−16.2%, every pair won, matvec stable ±5%) met the loop gate in its window, but 4/4 losses today means the win is not robust. Honest call: revert to the proven per-head-inside loop; parity re-confirmed after revert (169.1 vs 165.4 ms, +2.2%, ids identical — same code, same speed). The spike recipe (`idea-invented-grouped-hoist-proceed.md`) stands as spike-validated; any future port needs a quiet-box best-of-5 with balanced order before landing.

## 4. Lessons for the loop

- Order-balance every A/B (the landing run always ran baseline first — a confound that today's mixed-order check eliminates as an excuse, but the discipline should be default).
- Record matvec alongside attn in every pair: identical-code matvec drift is the in-situ noise meter. When it exceeds ±10%, timing claims are suspended, identity claims stand.
- Keep-grade requires a same-day re-verify on a second box state; big effects (≥1.5x) verify anywhere, small effects (≤20%) verify only on quiet boxes.

(End of file.)
