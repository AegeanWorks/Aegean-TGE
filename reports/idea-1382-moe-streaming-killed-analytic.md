# IDEA-1382: NVMe Expert Streaming with Arena and Background Warm — KILLED WITHOUT SPIKE (no MoE; working set fits RAM 3x over)

- **Idea ID**: IDEA-1382 (dossier `scout-2026-11-03-kilo-4target.md`: janas-style paged expert streaming — 4KiB arena, per-expert slots, LRU + prefill-predicted warm, O_DIRECT threads; claim: 48GB MoE on 32GB RAM; falsifier: "kill if bench shows no delta", premise "working set always fits RAM ... overhead for ~zero gain")
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: no carrier and no need — both falsifier arms trip on inspection). No branch created, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff), stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 27 after this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Port an arena/LRU/fetch state machine over mmap GGUF so a bigger-than-RAM MoE
runs by streaming experts (per-expert slots, prefill-predicted warm, background
O_DIRECT fetch) without forking the file format.

## 2. Predicted gain — and why both premises fail here

- (a) No carrier: zero `moe|MoE|expert` matches in `src/` and `include/`.
  The decoder is documented "deliberately narrow" (Qwen2-family dense only,
  one token at a time); the loader reads dense Qwen2/Llama/Phi tensors; no
  model in the local zoo is MoE (all dense: Qwen2.5 0.5–1.5B, Llama-3.2-3B,
  Phi-3.5-mini, SmolLM2). There is no expert table, no gate, no per-expert
  slot to stream into.
- (b) No need: largest local weights file is 2.23 GB (Phi-3.5-mini Q4_K_M);
  reference Llama-3B is 1.88 GB; box has 16 GiB with the biggest KV
  (8192-token f32 on Llama-3B ≈ 0.5 GB) still leaving the working set fitting
  ~3x over. The dossier's own kill condition ("working set always fits RAM —
  overhead for ~zero gain") is satisfied today: an arena/LRU/O_DIRECT
  subsystem would add threads, locks, and fetch latency to save zero bytes.

## 3. Falsifier (from the dossier) — trips verbatim

"Kill if bench shows no delta": no MoE workload exists to bench (no model,
no code path), so no delta is achievable by construction — the overhead-only
side of the trade is all that remains.

## 4. File anchors (current `master`, read but untouched)

- `src/`, `include/` — zero MoE/expert matches (grep, this cycle).
- `include/tge/models/qwen2.hpp:101-107` — "deliberately narrow" decoder.
- Local zoo listing (§6) + `docs/measurement-log.md` machine table (16 GiB,
  single DIMM).

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Re-read the dossier against the current tree; confirmed both the carrier
   gap and the capacity headroom.
2. Gates for the record: build clean, `ctest` pass, `aikit.py check` 2/4
   (test + secrets pass; hygiene + guidelines fail pre-existing).
3. No thinking-set diff owed: same binary, zero behavior change.

## 6. Measured numbers (inventory, not benchmarks — nothing to bench)

| check | result |
|---|---|
| MoE/expert code in `src/`, `include/` | 0 matches |
| MoE-capable local models | 0 of 8 (all dense) |
| largest weights file | 2.23 GB (Phi-3.5-mini Q4_K_M) |
| reference model | 1.88 GB (Llama-3B Q4_K_M) |
| box RAM | 16 GiB → working set fits ~3x over |
| achievable streaming delta | none (no workload, no path) |

## 7. Reimplementation recipe (deliberately NOT recommended — kill-grade)

For the record: would require an MoE-capable loader + gate + per-expert
execution (a new model architecture), then the arena/LRU/fetch subsystem —
i.e. a different engine first. Do not build either half until a >RAM MoE
workload exists on this box. Revisit only then, with the dossier's bench
as the gate.

## 8. Verdict and why

**KILLED without spike.** An expert-streaming subsystem with no experts to
stream and no capacity pressure to relieve is pure overhead: the dossier
anticipates exactly this outcome in its own falsifier. The idea is correctly
scoped for a MoE-serving project; this tree serves dense sub-4B models that
fit RAM several times over.

## 9. Exact commands run and their results

- Grep MoE/expert over `src/`, `include/` → zero matches.
- Local zoo listing → 8 dense models, max 2.23 GB.
- `cmake --build --preset release` → 54/54; `ctest` → pass.
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits, no push.
