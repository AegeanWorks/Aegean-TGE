# IDEA-1386: Sign-Aware PTQ with K-quant-aware GGUF Export — KILLED WITHOUT SPIKE (no write path; inference-only engine)

- **Idea ID**: IDEA-1386 (dossier `scout-2026-11-03-kilo-4target.md`: auto-round-style calibration-data PTQ with per-tensor embedding/lm_head formats, K-quant block geometry, MoE adapter; "better offline recipe ... no engine change"; falsifier: "Auto-round GGUFs do not beat imatrix KL at equal t/s — just another exporter")
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: the mechanism is an offline export pipeline and this repo has no write path — nothing to attach a measurement to). No branch created, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff), stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 31 after this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Calibration-data-only PTQ producing stock GGUF files (per-tensor formats for
embedding/lm_head, K-quant block geometry, MoE adapter) that must read back
through mmap and beat imatrix-quantized files on fidelity KL at equal t/s.

## 2. Predicted gain — and why there is no carrier

Every operative verb is offline: calibrate, quantize, export, write GGUF.
The engine's GGUF surface is read-only by design (`src/models/gguf.cpp`
parses; no write/export/save/convert/imatrix path exists anywhere in `src/`;
no `export|quantize|convert|imatrix` CLI surface). The models it runs arrive
pre-quantized (imatrix Q4_K_M etc.); the engine never produces a weight file,
so a "better exporter" cannot be staged, measured, or falsified here — its
falsifier demands producing GGUFs and pricing them, both outside the repo
boundary. The dossier itself concedes the shape ("no engine change").

## 3. Falsifier (from the dossier) — inapplicable, which is the kill

"Auto-round GGUFs do not beat imatrix KL at equal t/s." There is no GGUF
producer in-tree to run the comparison with; the only honest record is that
the idea lives in a different project (offline recipe tooling), not this
engine. Treating it as an engine cycle would manufacture scope the loop
cannot close.

## 4. File anchors (current `master`, read but untouched)

- `src/models/gguf.cpp` — read-only loader (no write/export/save matches).
- `apps/cli/main.cpp` — subcommands: generate, fidelity, info, gguf (inspect),
  plan, bench {continuous, sessions, batch, mem, gemv, gemm}, rope-plot,
  dogfood. No export surface.
- `src/` grep for `write_gguf|export|imatrix|ofstream|quantize.*file`:
  zero matches.

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Verified the read-only boundary (greps above + CLI inventory).
2. Gates for the record: build clean, `ctest` pass, `aikit.py check` 2/4
   (test + secrets pass; hygiene + guidelines fail pre-existing).
3. No thinking-set diff owed: same binary, zero behavior change.

## 6. Measured numbers (inventory, not benchmarks — nothing to bench)

| check | result |
|---|---|
| GGUF write/export/convert paths | 0 in `src/`, 0 CLI subcommands |
| imatrix/calibration surface | none (weights arrive pre-quantized) |
| MoE adapter relevance | none (dense-only engine, 1382 report) |

## 7. Reimplementation recipe (deliberately NOT recommended — kill-grade)

For the record: would require a new offline tool (calibration harness +
K-quant-aware exporter + imatrix-comparison harness) — a separate project
with its own loop, not an engine spike. Do not build it here. If such a tool
ever exists, its output GGUFs are testable by the unmodified fidelity
harness (`tge fidelity` vs scalar oracle) with zero engine changes — that is
the correct interface, and it already ships.

## 8. Verdict and why

**KILLED without spike.** An offline recipe with no in-repo producer, no
consumer-side hook missing (fidelity already prices any GGUF handed to it),
and a falsifier that cannot run here. The right outcome is routing, not
research: this belongs to recipe tooling, and its products are already
measurable by the untouched engine.

## 9. Exact commands run and their results

- Grep write/export/imatrix surface → zero matches (carrier absent).
- CLI inventory → no export path.
- `cmake --build --preset release` → 54/54; `ctest` → pass.
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits, no push.
