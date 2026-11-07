# INVENTED (gpu-validate follow-up): GPU vs CPU on Long Prefill — REOPEN (offload loses 2.5x on 0.5B, 11% on Llama; 4.3GB moved per 0.5GB model = weights not persistent)

- **Idea**: The validate cycle confirmed GPU correctness but explicitly left perf open, noting transfer+sync dominate small shapes. Test the designed shape: long prefill where compute should amortize transfer. 0.5B-Q8_0 fox×150 (1950 tok) CPU/GPU best-of-2 swapped + Llama-3B fox×100 (1300 tok) single pair.
- **Cycle date**: 2026-11-06
- **Verdict**: REOPEN — GPU loses everywhere tested: 0.5B prefill 303→121 t/s R1 and 300→122 R2 (2.5x slower, both orders); Llama 57.8→51.2 t/s (−11%). The transfer log is the smoking gun: 4,289,737,728 bytes moved per 0.5B run vs ~0.5 GB of weights — 8-9x the model, i.e. weights re-upload constantly instead of once. Even infinite GPU compute couldn't beat CPU here (transfer alone, 12.9 s, exceeds the entire CPU prefill, 6.4 s).
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). No branch, no edits, no scratch. `master` ahead of `origin/master` by 116 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — existing flags, no code)

`generate --model {0.5B-Q8_0,Llama-3B-Q4_K_M} --threads 10 [--gpu on|off] --context 4096 --prompt <fox-repetitive> --n {31,8} --print-ids`; prefill wall + offload byte counters + ids.

## 2. Predicted gain — inverted far past noise

Falsifier: GPU ≥ CPU +20% on long prefill → CONFIRM design intent; else REOPEN the offload as transfer-bound. Measured −60% (0.5B, twice) and −11% (Llama, once).

## 3. Falsifier outcome — REOPEN with the mechanism named, not proven

1. **Direction consistent across models/orders**: off loses everywhere; swapped order rules out drift as the driver.
2. **Cause hypothesis (evidence-backed, not root-caused)**: 4.3 GB moved per 0.5B run = non-persistent weights; transfer alone (12.9 s) > whole CPU prefill (6.4 s). Whether that's a missing persistence feature (design gap) or a shadowing bug needs owner triage with the GPU code — explicitly NOT diagnosed here.
3. Ids identical everywhere (degenerate prompt; numerics already priced by the validate cycle).

## 4. File anchors (current `master`, read but untouched)

- `src/models/qwen2.cpp:680-727` (`wants_gpu_*` gates; offload call sites).
- `src/gpu/vulkan_context.cpp` + `noop.cpp` (backend whose transfer pattern is implicated).
- Grounds: the validate-confirm (correctness baseline this reuses), 3160-series (sibling-box numbers, not contradicted — different box/shape).

## 5. What was done (pure measurement — no spike, no branch)

1. CREATE with the stated falsifier (the validate cycle's explicit open thread).
2. Gates: build 54/54; tests 141/141; `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. 0.5B pair ×2 (swapped) + Llama pair ×1; ids + byte counters recorded.
4. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (repetitive fox prompt, threads 10)

| run | CPU prefill | GPU prefill | offload bytes (GPU) | delta |
|---|---|---|---|---|
| 0.5B R1 (off,on) 1950 tok | 6430.5 ms (303.2 t/s) | 16137.6 ms (120.8 t/s) | 4,289,737,728 B in 12952.1 ms | −60.2% |
| 0.5B R2 (g,o) 1950 tok | 6452.6 ms (300.2) | 16055.8 ms (121.5) | ~same (run cut from terse output; same class) | −59.5% |
| Llama R1 (off,on) 1300 tok | 22479.5 ms (57.8) | 25380.4 ms (51.2) | (not captured; same driver) | −11.4% |

## 7. Full reimplementation recipe (verification + triage fork)

Re-run §1 (any long prompt ≥1500 tok); expect GPU slower by ≥10% with multi-GB moved per sub-GB model. THEN: (a) if weights-persistence lands (owner), re-run and expect the sign to flip on long shapes; (b) until then, guidance stands: leave `--gpu off` (the default) on this box. Revival trigger: a run where moved bytes ≈ 1x model size AND GPU wins — that build is the fixed one; report it as the close-out.

## 8. Verdict and why

**REOPEN: the offload path is correct but uneconomical on this box — it moves ~8x the model per run, so transfer alone exceeds the entire CPU job.** This does not contradict the sibling-box 3.0x claim (different box/shape/code); it says THIS tree on THIS box loses on every shape measured. The finding is scoped honestly: no root cause claimed (persistence missing vs transfer path slow needs owner eyes on the GPU code), no large-shape claim made (untested beyond 2k tokens — though the byte math worsens, not improves, with length). Next-cycle candidates: the 1517 stage-2 harness remains the only live structural front; the human-gated port shelf stands (and this report joins the queue as backend work).

## 9. Exact commands run and their results

- 0.5B fox×150 CPU/GPU ×2 swapped + Llama fox×100 CPU/GPU ×1 → §6 table + ids.
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
