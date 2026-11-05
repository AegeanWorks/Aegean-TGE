# IDEA-1384: Paged Quantized KV with CoW Prefix Sharing — SPLIT (zero-copy killed analytic; quantized snapshot keep-grade recipe)

- **Idea ID**: IDEA-1384 (dossier `scout-2026-11-03-kilo-4target.md`: quant.cpp paged blocks, per-head pointers, CoW share_block, TurboQuant framing; falsifier: "take CoW design only, kill if no win vs 4-slot bench continuous")
- **Cycle date**: 2026-11-06
- **Verdict**: SPLIT. (a) Zero-copy block sharing vs the shipped memcpy CoW: killed analytic — saves only the memcpy sliver (~0.3 ms) of a win dominated by skipped recompute (~400 ms), plus zero memory saving (slots preallocated). (b) Quantized snapshot (f16+i8 CoW): spiked, measured, KEEP-GRADE — 4.4–5.0x prefill-side wins with cold-vs-hit FNV MATCH in all tiers on both models, ids identical. Spike fully reverted per loop rules; the full recipe below is the keep. Only this report committed.
- **Tree state at close**: `git status` clean (no diff), spike branch created and deleted with nothing committed, stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 25 after this report. Nothing pushed (human gate).

## 1. Mechanism

(a) Zero-copy per-head block pointers into shared prefix blocks instead of
memcpy-ing rows per sequence. (b) Extend the shipped f32-only cross-sequence
snapshot to f16 and i8 tiers: snapshot kind-tagged rows (+scales for i8),
hit path copies snapshot rows verbatim and skips recompute — byte-identical
by construction (deterministic quantize of identical rows), same gate as f32
(≥16 common tokens, longer prompt, verifier excluded).

## 2. Predicted gain (with the math)

Shared-prefix prefill of C tokens costs full matvecs per sequence; a snapshot
hit recomputes only the (count − C) tail: expected win ≈ C/(count−C) bounded
by overhead — 120/(32) ≈ 3.75x theory on the 8x(120+32) shape; shipped f32
measured 3.06–3.32x. Zero-copy would additionally save the memcpy:
C × layers × kv_dim × bytes/row — 120×24×128×8 B ≈ 5.9 MB ≈ 0.3 ms on 0.5B
vs ~400 ms recompute saved: <0.1% of the win, 0% end-to-end, 0 bytes saved
(slots preallocated regardless).

## 3. Falsifier (from the dossier, staged)

Kill zero-copy if its saving is a sliver of the recompute win (trips: 0.3 ms
vs 400 ms). Quantized snapshot: kill if prefill-side win <2x on the replay
workload in any tier, or any cowgate MISMATCH, or any ids divergence. All
quantized arms pass with margin.

## 4. File anchors (current `master`, read before writing)

- `src/models/qwen2.cpp:34-55` (`SharedPrefix`, f32-only + comment);
  `:608-616` (load clear); `:1819-1867` (HIT, f32 memcpy); `:1996-2033`
  (POPULATE); `:1752-1816` (`prefill_run`, H1 same-sequence record +
  chunk loop honoring `consumed`).
- `include/tge/models/qwen2.hpp` (cache members; `reset_cache`, `argmax`).
- `apps/cli/main.cpp` (`command_generate` prefill print; bench continuous
  uses `forward_batch`, which has NO CoW consult — named follow-up, §8).
- Measurement-log §2026-11-05 (f32 CoW: 3.06–3.32x prefill-side, FNV 7/7).

## 5. What was done (half A analytic; half B temporary spike, fully reverted)

1. `git checkout -b spike-1384-qcow` (no commit ever made on it).
2. Analytic half A from the shipped numbers + memcpy math (§2/§6).
3. Spike half B (3 files, ~220 lines): snapshot tiers (`k_half/v_half`,
   `k_q8/v_q8` + scales, `kv_kind`), load-clear extension, kind-aware HIT
   (verbatim row+scale memcpy per tier) and POPULATE (reads tier caches
   post-hoc — no store-loop changes), `kv_fingerprint` (FNV-1a over the
   active tier) + `clear_shared_prefix` spike gates, env-gated
   `TGE_COW_REPLAY` three-stage replay driver (P populate, P+A H1 path, P+B
   CoW HIT; then hash-hit vs forced-cold hash + MATCH print, exit(0)).
4. `cmake --build --preset release` clean; `ctest` pass; 141/141 (no new
   unit tests — CLI-observable behavior; suite guards regressions).
5. Replay A/B: f32 control (reproduces shipped win), i8 best-of-3, f16,
   Llama-3B i8 confirmation; cowgate MATCH everywhere; normal-generate ids
   parity spike-vs-baseline (f32 + i8).
6. Reverted all 3 files, verified empty diff, deleted branch. Final gates on
   master: build clean, `ctest` pass, `aikit.py check` 2/4 (test + secrets
   pass; hygiene + guidelines fail pre-existing).

## 6. Measured numbers (Qwen2.5-0.5B-Q8_0 + Llama-3.2-3B-Q4_K_M, `--gpu off`, warm page cache)

157-token shared prefix + 32-token tail (cap 256; H1 path measured as reference):

| model / tier | cold prefill (157 tok) | CoW-hit replay (189 tok) | prefill-side win | cowgate |
|---|---|---|---|---|
| 0.5B f32 (control) | ~500 ms | ~100–115 ms | **~4.6x** | MATCH ×3 |
| 0.5B i8 (spike) | ~494 ms | ~98–105 ms | **~4.8–5.0x** | MATCH ×3 (identical hashes all runs) |
| 0.5B f16 (spike) | 476 ms | 108 ms | **~4.4x** | MATCH |
| Llama-3B i8 (spike) | 2365 ms | 535 ms | **~4.4x** | MATCH |

H1 same-sequence path (pre-existing, tier-agnostic): ~100 ms on the same
shape — the snapshot HIT matches H1's skip rate, as constructed (same
`consumed` mechanism downstream).
ids: spike f32 fox ids byte-identical to recorded baseline
(`576 ... 1854 13`); spike i8 fox ids byte-identical to rebuilt-baseline i8
ids; haiku pair identical. No fidelity run owed: hit rows are verbatim copies
of cold rows (memcpy, no arithmetic) — same evidence class as the shipped
f32 CoW (FNV-gated, no suite test then either).

## 7. Full reimplementation recipe (keep-grade for half B; half A stays dead)

1. `SharedPrefix` (file-local, qwen2.cpp): add `k_half/v_half`,
   `k_q8/v_q8`, `k_q8_scales/v_q8_scales`, `int kv_kind = 0`; clear all in
   the load-clear block.
2. HIT: gate on `kv_kind == (kv_half_ ? 1 : kv_q8_ ? 2 : 0)` instead of the
   f32-only gate; per-kind verbatim memcpy (f32 K/V; f16 halves; i8 rows +
   `dst/32`, `src/32` scale blocks of kv_dim/32 floats).
3. POPULATE: kind-tag the snapshot; allocate per-kind vectors
   (`cells = layers*cap*kv_dim`, q8 scales `cells/32`); copy tier cache rows
   (+scales) post-hoc. Keep the ≥16-token floor, cap, verifier exclusion.
4. Driver + gates (spike-only, or a `bench cow` later): env-gated
   populate → H1 → HIT replay with phase prints; `kv_fingerprint`
   (FNV-1a over active tier incl. scales) + `clear_shared_prefix` for the
   MATCH gate; ids parity via normal generate.
5. Do NOT build: zero-copy block pointers (saves the 0.3 ms memcpy only),
   paged BlockTable wiring (separate epic — see §8).
6. Named follow-up (not this cycle): port the CoW consult into
   `forward_batch`'s mixed batches (per-sequence chunk-skipping) so
   continuous batching admissions share prefixes — that is where the
   dossier's "4-slot wins" would actually materialize; the per-admission
   win magnitude is established here (~5x of the shared part).

## 8. Verdict and why

**SPLIT.** Zero-copy is correctly dead: it optimizes the 0.3 ms memcpy inside
a 400 ms win while saving zero bytes — the dossier's "take CoW design only"
was right, and CoW ships. The quantized snapshot is correctly alive:
4.4–5.0x prefill-side in f16+i8 with cold-vs-hit FNV MATCH on both models —
every falsifier arm passes, so the portable half of 1384 is done as a
recipe. Revisit zero-copy never (arithmetic forecloses it); revisit the
forward_batch port when continuous-batching prefix sharing is scheduled —
the per-admission payoff is now priced.

## 9. Exact commands run and their results

- `git checkout -b spike-1384-qcow` → new branch.
- Spike (struct + load + hit + populate + fingerprint + driver).
- `cmake --build --preset release` → 54/54; `ctest` → pass, 141/141.
- f32 replay control ×3 (validates driver vs shipped 3.06x: got ~4.6x).
- i8 replay ×3 (all MATCH, ~5x); f16 replay (MATCH, ~4.4x); Llama-3B i8
  replay (MATCH, ~4.4x).
- Saved spike exe; normal-generate ids spike vs baseline (f32 + i8,
  identical); stashed; rebuilt baseline for the i8 ids pair.
- `git stash pop`; `git checkout -- <3 files>` → diff empty;
  `git checkout master`; `git branch -D spike-1384-qcow`.
- Final: build clean, `ctest` pass, `aikit.py check` 2/4 (pre-existing fails).
- No push.
