# INVENTED: kq8d head + speculative decoding — KILLED (verify path is float-only by design; head scope unreachable)

- **Idea**: Combine `--speculative` with the kq8d LM-head scope, on the theory that each verifier forward still pays the 308 MiB head and would benefit from the integer row kernel.
- **Cycle date**: 2026-11-08
- **Verdict**: **KILLED — architectural mismatch.** In full-accept speculative runs the head timer records **0 ms**: the verifier's output projection goes through `verify_draft_logits` → `batch_matvec(..., float_only=true)`, which deliberately bypasses the integer path for occupancy-invariant math. The `TGE_KQ8D_HEAD` flag only reaches the rare 1-row correction forward, so the two A/B pairs (0.939x, 1.235x) disagree and measure box drift, not the scope.
- **Tree state**: spike reverted from `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp` and `apps/cli/main.cpp` (temporary head timer); 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Mechanism and what actually runs

Speculative decoding uses batched verify forwards (`verify_draft_logits`), not the single-token `forward`. The verifier's output projection is routed as:

```text
batch_matvec(output_, batch_h_, logits_batch_, count, q_out, /*float_only=*/true);
```

`float_only` forces the float-dequant path so mixed decode/prompt batches keep occupancy-invariant math (the continuous-batch LEAK near-tie rule). The kq8d head branch lives in `Qwen2::matvec` (single-row), which is called only by `output_logits` — used by the correction forward after a partial accept, which does not occur in a full-accept stream.

A temporary `phase_head_ns_` timer wrapping `output_logits` confirms it: **0.0 ms** over both speculative runs, versus hundreds of ms in non-speculative decode.

## 2. Falsifier

- **KEEP** if the head scope is actually reached by the verify forwards and decode gains with identical ids.
- **KILL** if the verify path bypasses the scope.

Measured: 0 ms head time → unreachable; kill.

## 3. Measured numbers

### Speculative 4096-token periodic prompt, n=64, f32, 6 threads

| pair | spec | spec + head kq8d | ratio | head timer | ids |
|---:|---:|---:|---:|---:|---|
| 1 | 11.01 t/s | 10.34 t/s | 0.939x | 0 → 0 ms | IDENTICAL |
| 2 | 9.93 t/s | 12.26 t/s | 1.235x | 0 → 0 ms | IDENTICAL |

The ratios disagree in direction, and the head timer is zero in both arms: no head work is executed, only window drift.

Spec decode phase split in these runs: **matvec 35–38%, attention 60–63%** (speculation pins the per-head schedule), norm ~2%. Even if the head path applied, the head is a fraction of the 35% matvec share, so the ceiling would be a few percent.

## 4. Interpretation

- The speculative verifier intentionally runs the float-only batch path; making the head integer there would change the arithmetic of every verified row and risks the near-tie occupancy invariants the `float_only` flag exists to protect. It is not a drop-in flag flip.
- With speculation on, attention (per-head, long context) dominates decode at 60%, so the head matvec is a smaller lever than in baseline decode.
- The cycle-43 result stands: speculation is a ~2x mode on predictable prompts. The kq8d head remains a baseline-decode win; the two do not compose.

## 5. Disposition

- **Do not combine** the kq8d head with speculative verify under the current invariants.
- If the head tier is ever wanted inside speculation, it needs a dedicated verify-head path with its own fidelity/ids gate and a replacement for the occupancy-invariance guarantee — a separate, larger design decision.
- Keep `--speculative` as the primary lever for predictable prompts and the kq8d head for baseline decode.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_KQ8D_HEAD=0|1 tge generate --tokens <4096 periodic ids> --n 64 \
  --threads 6 --gpu off --kv f32 --context 4352 --speculative --print-ids
git checkout -- include/tge/models/qwen2.hpp src/models/qwen2.cpp apps/cli/main.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
