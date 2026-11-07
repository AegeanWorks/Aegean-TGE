# INVENTED: GPU Offload Validation — CONFIRM (ids identical, fidelity at dense floor both modes; slower on small shapes as designed)

- **Idea**: The GPU path (prefill batch + output GEMV offload on Q8_0) runs in this build — the load line says ready, and `--gpu on` executes offloads — but NO report validates it on this tree: all prior GPU work closed on a sibling box, and the stuck items assumed no SDK. Run the fixed thinking set + a fidelity pair, CPU vs GPU, on 0.5B-Q8_0 (the config the offload serves).
- **Cycle date**: 2026-11-06
- **Verdict**: CONFIRM — the GPU path is numerically sound. Story + creative ids byte-identical CPU-vs-GPU (32/32, 24/24). Fidelity arch/199-pos: CPU KL 1.080e-11 / 199-199 vs GPU KL 1.129e-11 / 199-199 — same floor (5% relative on 1e-11 is kernel-order noise), zero flips either way.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). No branch, no edits, no scratch files. `master` ahead of `origin/master` by 116 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — existing flags, no code)

`generate --model <0.5B-Q8_0> --threads 6 [--gpu on|off] --prompt <story|creative> --n {32,24} --print-ids`; `fidelity --corpus docs/architecture.md --context 512 --tokens 200 --threads 6 [--gpu on|off]`.

## 2. Predicted gain — a soundness verdict, landed

Falsifier: identical ids + fidelity at the dense floor on both modes → CONFIRM the shipped path; any divergence/crash → REOPEN as a backend bug (would have been the first on-record defect of the GPU path on this tree). Measured: identical everywhere that matters.

## 3. Falsifier outcome — CONFIRM

1. **Identity**: story + creative ids byte-identical CPU-vs-GPU.
2. **Numerics**: fidelity digit-class identical (KL 1.08e-11 vs 1.13e-11, both 199/199) — the 4% KL delta is AVX2-vs-GPU reduction-order noise at 1e-11, not a signal.
3. **Perf (observed, NOT gated — single runs)**: story decode 85.3 t/s CPU vs 46.9 t/s GPU; the offload log shows the why (512.8 ms prefill + 255.1 ms output transfer+sync on a 32-token shape). Transfer+sync dominate small interactive shapes — consistent with the single-sync design, and explicitly NOT a verdict on large-batch prefill (unmeasured here, do not cite this cycle for that).

## 4. File anchors (current `master`, read but untouched)

- `src/models/qwen2.cpp:680-727` (`wants_gpu_*` gates: Q8_0/F32 only, `clean` = no VNNI/scalar/non-auto-attention; output offload call).
- `src/gpu/vulkan_context.cpp` + `noop.cpp` (compiled backend, `use_gpu()` gate).
- Grounds: the stuck GPU items (assumed no SDK — superseded on this box: the SDK is here and the path runs), 3160-series (sibling-box numbers, not this tree).

## 5. What was done (pure measurement — no spike, no branch)

1. ANALYZE: confirmed offload executes on this box (not just links); confirmed the full prior GPU record is sibling-box (nothing validates THIS tree's backend).
2. Gates: build 54/54; tests 141/141; `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. Thinking set both modes (ids); fidelity pair both modes (§3).
4. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (0.5B-Q8_0, threads 6)

| check | CPU | GPU (`--gpu on`) |
|---|---|---|
| story ids (32) / creative (24) | baseline | byte-identical both |
| fidelity KL mean / top-1 | 1.080e-11 / 199-199 | 1.129e-11 / 199-199 |
| story decode (single runs, context) | 85.3 t/s | 46.9 t/s (offload: 512.8 + 255.1 ms transfer/sync) |

## 7. Full reimplementation recipe (verification — re-run to re-verify)

Re-run §1; expect identical ids and floor-level fidelity on both modes. Revival trigger: any CPU/GPU id divergence, any non-floor fidelity on GPU, or a crash — file as a GPU-backend bug with the exact flag set (this cycle establishes the clean baseline it would regress from). Open (not this cycle): large-batch prefill GPU-vs-CPU timing, which is the shape the offload is designed for.

## 8. Verdict and why

**CONFIRM: the GPU backend on this box is numerically sound and its small-shape perf profile matches its design (transfer-bound).** This closes the validation hole the stuck items left open — but only the correctness half; no throughput claims are made here. Next-cycle candidates: the 1517 stage-2 harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- Thinking set ×2 modes → identical ids both prompts.
- Fidelity arch pair, CPU vs GPU → §6 row.
- Story totals noted for context (single runs, not gated).
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
