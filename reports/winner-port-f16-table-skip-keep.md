# WINNER PORT: f16 Prefix Table + Skip — LANDED, keep-grade (+57% f16 bench, tier-tagged)

- **Source**: invented follow-up to winners #2–#4 + #6 (`--kv auto` routes long ctx to f16, which disabled the whole prefix apparatus; this port reunites them). No separate dossier — the mechanism is proven by the same falsifiers.
- **Port commit**: `5e5fcae` (`perf(aegean-tge): multi-prefix table and batch skip span f16 KV (tier-tagged)`), 1 file (`src/models/qwen2.cpp`, +87/−45). Temp f16 MATCH driver reverted before commit. Nothing pushed (human gate).
- **Verdict**: WINNER — kept. The f16 long-context path now skips like f32.

## 1. What landed

Tier tag (`bool half`) on every table entry, set at populate, required to match on every consult (HIT/scan/prime/shared_prefix_len/S1 re-scan); copy helpers (`table_store/‌load_kv_row`) that memcpy f32 or cvt f16 traffic with bit-exact roundtrips; all five f32-only gates relaxed to `!kv_q8` (i8 still excluded — no strided i8/table i8 kernel). The critical correctness point: an f32-populated slot must never serve an f16 run and vice versa (their K/V differ by the tier's rounding), so the tag — not just geometry — gates every path.

## 2. Remeasure (this tree; baseline = pre-port binary, interleaved best-of-3)

Continuous 8×32 P=32 `--kv f16`, Q8_0 0.5B, threads 6:

| pair | baseline | port | delta |
|---|---|---|---|
| 1 | 77.89 t/s | 138.35 t/s | +77.6% |
| 2 | 109.30 | 171.70 | +57.1% |
| 3 | 107.77 | 128.30 | +19.0% |

Best-of-3: **109.30 → 171.70 t/s (+57%)**; every pair won; transcripts byte-identical; PRIME fires on all f16 admissions, zero EVICT. f32 no-regression check: 142.64 vs 135.22 (−5.2%) then 169.74 vs 174.11 (+2.6%) — no direction, i.e. the helper refactor is free. kv-auto integration: `--kv auto` (ctx 4096 → f16) populates + primes + skips, 133.55 t/s, transcripts identical.

MATCH (temp driver, reverted): batch-populate f16 → prefill HIT → final-logits memcmp MATCH vs cold f16 prefill; control without the path MATCHes trivially.

## 3. Gates

Build 54/54, `tge_tests` 145/145 (port + final tree, no new committed tests — the MATCH driver was the differential proof, reverted per spike discipline). Fidelity `--kv f16` on port: 2.142e-06 / 1022-1023 — digit-identical to pre-port (table excluded from the all_logits path, as designed). Thinking set: story ids byte-identical on f16 and f32. `aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing).

## 4. Why it stays

Same falsifier as S2 (+15% with identical transcripts) passed on best, median (+57%), and every pair (+77/+57/+19). The tier-tag design makes cross-tier contamination impossible by construction, and the MATCH + fidelity numbers confirm zero new numerics.

## 5. Next winner in queue

kv follow-ups done (flag + f16 span). Remaining: 1517 follow-ups (done), quiet-box items (hoist re-verify round 3 needs a layout hypothesis first; thread sweep needs a cool box), captured-real-activation decode reopen (needs the capture harness).

(End of file.)
