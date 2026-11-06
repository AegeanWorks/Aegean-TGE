# IDEA-1518: NVMe-to-RAM MoE Expert Streaming with 8-bit KV + Draft Routing — KILLED WITHOUT SPIKE (no MoE; other halves ship)

- **Idea ID**: IDEA-1518 (dossier `goal-sweep-2026-11-03.md`: janas-style O_DIRECT + io_uring expert streaming, per-layer LRU, prefetch, 2MB huge pages + 8-bit KV + draft routing; 80B-class MoE 26.4 vs 3.0 tok/s; falsifier: bench tge model vs janas file on the same CPU — fail on decode t/s + peak RSS regression, top-1 below parity, or disk-stall without NVMe)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: the MoE half has no carrier — same kill as 1382 — and the falsifier's own bench requires "one tge MoE file", of which zero exist; the 8-bit KV and draft-routing halves already ship). No branch created, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff), stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 32 after this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Stream routed experts NVMe→RAM (mmap-adjacent O_DIRECT), per-layer LRU +
prefetch, 8-bit KV, draft-routed decode for MoE on small-RAM boxes.

## 2. Predicted gain — and why each third fails or ships

- MoE streaming: no carrier. Zero `moe|MoE|expert` matches in `src/` +
  `include/` (1382 report); zero MoE models locally (all dense ≤2.23 GB);
  box RAM 16 GiB fits every working set ~3x over. The falsifier's bench
  ("one tge MoE file") cannot be staged — there is no file on either side.
- 8-bit KV parity: SHIPS — i8 tier priced (KL 3.2e-4, top-1 kept), kq8
  integer prefill default-on (+36%), `--kv i8` on every relevant CLI.
- Draft routing: SHIPS — SKIP-lock + prompt_lookup family + fused verify,
  acceptance-gated with identical ids.

## 3. Falsifier (from the dossier) — cannot trip, which is the kill

Decode t/s + RSS regression, top-1 below parity, or disk-stall without NVMe
all presuppose an MoE workload that does not exist here. A falsifier whose
preconditions are unstageable kills the idea, not the tree.

## 4. File anchors (current `master`, read but untouched)

- `reports/idea-1382-moe-streaming-killed-analytic.md` — the carrier kill
  this one extends.
- `src/models/qwen2.cpp` (`kv_q8_` paths), `apps/cli/main.cpp`
  (`--kv i8`, `--speculative`) — the shipped halves.
- Local zoo: 8 dense models, max 2.23 GB.

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Mapped each third to dead (MoE) or shipped (i8 KV, drafters).
2. Gates for the record: build clean, `ctest` pass, `aikit.py check` 2/4
   (test + secrets pass; hygiene + guidelines fail pre-existing).
3. No thinking-set diff owed: same binary, zero behavior change.

## 6. Measured numbers (inventory — the bench is unstageable)

| check | result |
|---|---|
| MoE code/models | 0 / 0 |
| i8 KV | ships (KL 3.2e-4, +36% prefill) |
| draft routing | ships (5 mechanisms, ids identical) |

## 7. Reimplementation recipe (deliberately NOT recommended — kill-grade)

Would require an MoE-capable model + loader + gate + the streaming
subsystem first — a different engine. Do not build. The i8-KV and
draft-routing thirds need no recipe (shipped; reports `ee401e5`,
`59e22f6` family).

## 8. Verdict and why

**KILLED without spike.** Two shipped thirds plus a MoE third with no file,
no code, and no RAM pressure. The dossier's falsifier is unstageable by
construction here.

## 9. Exact commands run and their results

- Grep MoE surface → zero (1382 record stands).
- `cmake --build --preset release` → 54/54; `ctest` → pass.
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits, no push.
