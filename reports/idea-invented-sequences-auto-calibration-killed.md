# INVENTED: Short 4-vs-8 calibration for `--sequences auto` — KILLED (opposite choices within minutes; variance > signal)

- **Idea**: After the live-memory `--sequences auto` load, run a short throughput calibration (4 vs max active slots) on the real `forward_batch` path, then use the faster count for the measured run. This replaces the CPU-busy guard killed in cycle 14.
- **Cycle date**: 2026-11-07
- **Verdict**: **KILLED.** A 6-step calibration chose 4 (16.1 vs 9.7 t/s); an 8-step version minutes earlier chose 8 (18.1 vs 22.7 t/s). The 4-vs-8 difference on this box is smaller than its run-to-run variance, so a short sample cannot decide. The longer version also cost 13–18 s — not a startup-cheap calibration.
- **Tree state**: spike reverted from `apps/cli/main.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; nothing pushed.

## 1. Mechanism

The auto path resolves max slots from live memory as in cycle 13. The spike then:

1. warms the weight stream with one `measure(max)` arm;
2. measures 4 active slots and max active slots on the real `forward_batch` path;
3. each arm decodes a fixed token at positions 0..steps, with 6 or 8 timed steps;
4. returns **tokens/s** (`steps × active / seconds`);
5. chooses max if `max > 4 × 1.02`, else 4;
6. runs the measured benchmark with the chosen active count (the model was loaded with max slots, so 4 uses slots 0–3).

The first implementation returned **steps/s**, which systematically favored 4 because an 8-slot step decodes 8 tokens. That metric bug was fixed before the reported runs; it is a required part of the recipe.

## 2. Falsifier

- **KEEP** only if the calibration picks the faster arm repeatably and the resulting auto run is at least fixed 4 on the 8×32 load.
- **KILL** if two short runs disagree, or the calibration costs as much as the run.

Measured: two runs disagreed; kill.

## 3. Measured numbers

Llama-3.2-3B-Q4_K_M, `bench continuous --requests 8 --tokens 32 --threads 6 --kv f16`.

### 8-step, warmup + 2 interleaved rounds (5 arms)

```text
auto  : max 8 slots (live 11.4 GiB, usable 9.4 GiB, headroom 7.5 GiB)
calib : 4 slots 18.1 t/s, max 8 slots 22.7 t/s -> 8
load  : 19033 ms (includes calibration)
decode: 248 tokens, 17.90 t/s
```

Calibration cost ~13–18 s; it chose 8.

### 6-step, warmup + one 4 + one max arm (3 arms)

```text
auto  : max 8 slots (live 12.1 GiB, usable 10.2 GiB, headroom 8.3 GiB)
calib : 4 slots 16.1 t/s, max 8 slots 9.7 t/s -> 4
load  : 11058 ms (includes calibration)
decode: 248 tokens, 6.53 t/s
```

Minutes later, same binary/model/box, the short calibration chose 4. The main 4-slot decode also ran at 6.53 t/s, so the window was slow; the point is that the short arms did not measure a stable 4-vs-8 ordering.

### Context from the prior cycles

- Cycle 11: 8 vs 4 median 1.337x, one 0.790 loss.
- Cycle 12: 1.253x median, 3/3.
- Cycle 13: 1.104x median, one 0.704 loss.
- Cycle 14: 8 slots beat guarded 4 under synthetic CPU load by 28%.

The 4-vs-8 ratio ranges from 0.70 to 1.87 across runs; a 3–5-arm 6–8-step calibration is inside that noise band. A reliable calibration needs a longer ABBA measurement (≥16 steps, 2+ rounds), which costs about as much as the measured run.

### Fidelity (unchanged)

| KV | mean KL | top-1 |
|---|---:|---:|
| f32 | 1.130e-11 | 63/63 |
| f16 | 2.501e-07 | 63/63 |
| i8 | 2.373e-04 | 61/63 |

The spike changes only CLI slot selection.

## 4. Why it failed

- **Signal smaller than box variance.** The same 8×32 benchmark varies by ±30–90% run to run; a 6–8-step arm samples a tiny slice of that.
- **Order/thermal bias.** The 4 arm runs before the max arm (or vice versa); the box's load changes within seconds. Without ABBA and a longer window, the ordering dominates.
- **Cost.** The 8-step 5-arm version added 13–18 s, comparable to the main run; not a startup-cheap calibration.
- **Metric trap.** Comparing steps/s instead of tokens/s would have made the 8-slot arm look slower by construction; the recipe must use tokens/s.

## 5. Reimplementation recipe (kill-grade, for the record)

1. Load with max slots from the live-memory policy (cycle 13 recipe).
2. `measure(active)`:
   - vectors of `active` tokens (128), positions (0), sequences (0..active-1);
   - one warmup `forward_batch` at position 0;
   - time `kCalibSteps` `forward_batch` calls at positions 1..k;
   - return `kCalibSteps * active / seconds` (tokens/s).
3. Warmup arm `measure(max)`; then `measure(4)` and `measure(max)`; choose max if `many > four * 1.02`.
4. Print the three rates and the chosen count; run the main bench with that active count.
5. Gate: repeat the calibration at least twice and require the same winner; if not, do not use it.
6. Expected from this cycle: 8-step chose 8, 6-step chose 4 → do not ship a short calibration.

## 6. Verdict and disposition

**KILLED.** Do not add a short calibration. Keep `--sequences auto` as the live-memory-only opt-in from cycle 13; it is safe but not a throughput promise. If a reliable slot decision is needed, use a full 4-vs-8 benchmark or a fixed preset. A future calibration must be longer, ABBA-ordered, and budgeted as a benchmark, not as startup logic.

## 7. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
tge bench continuous --model Llama-3B --requests 8 --tokens 32 --threads 6 --sequences auto --kv f16
# first: calib 18.1 vs 22.7 -> 8; second: calib 16.1 vs 9.7 -> 4
tge fidelity ... --kv f32|f16|i8
git checkout -- apps/cli/main.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
