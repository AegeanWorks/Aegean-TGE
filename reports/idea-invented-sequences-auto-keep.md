# INVENTED: Budget-aware `--sequences auto` — KEEP as opt-in (auto picks 8, +25% median aggregate)

- **Idea**: Add `--sequences auto` to `bench continuous`. Instead of a fixed 4 slots, resolve the slot count from the model's `ModelPlan` and the real `MemoryBudget`: `place(plan, usable, context, 8)` gives `max_sessions`; clamp to 1–8, prefer 4, warn when the budget cannot fit 4. This captures the 8-slot throughput win only when the budget supports it.
- **Cycle date**: 2026-11-07
- **Verdict**: **KEEP as an opt-in.** Auto resolved to **8 slots** in all runs (usable 11.2–12.8 GiB, KV 896 MiB/session) and beat fixed 4 in **3/3 paired runs**: ratios 1.253, 1.391, 1.252; median **1.253x**. First transcripts identical; fidelity tiers unchanged. It is not made the default because the budget check uses total usable memory, not current page-cache headroom; the cycle-11 8-slot loss remains a real pressure case.
- **Tree state**: spike reverted from `apps/cli/main.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; nothing pushed.

## 1. Mechanism

The engine already supports `--sequences N`; the only missing piece was a policy that picks N from memory. The spike:

1. parses `--sequences auto` as a sentinel (`0`);
2. before load, opens the GGUF metadata-only (`File::open(path, false)`);
3. builds `ModelPlan` with `analyse(file, context)`;
4. reads `MemoryBudget{}.snapshot().usable`;
5. calls `place(plan, usable, plan.context, 8)`;
6. uses `std::clamp(decision.max_sessions, 1, 8)`;
7. prints `auto : N slots (usable X GiB, KV/session Y MiB)`;
8. if `N < 4`, prints a warning but uses N rather than overcommitting; a strict 4-floor can be a policy flag if wanted.

No engine code changes; the optimization is the slot-count policy.

## 2. Predicted gain / falsifier

Cycle 11 measured 8 f16 slots at a 1.337x median over 4, with a memory-pressure loss. A budget-aware pick should keep the win when 8 fits and fall back when it does not.

- **KEEP as opt-in** if auto is ≥1.15x fixed 4 on the 8×32 load, transcripts identical, and it never exceeds 8.
- **Do not default** if the budget can still pick 8 under pressure and lose; the flag stays opt-in.

## 3. Measured numbers

Llama-3.2-3B-Q4_K_M, `bench continuous --requests 8 --tokens 32 --threads 6 --kv f16`, context 4096, 248 decode tokens.

| pair | fixed 4 | auto | ratio | auto choice |
|---:|---:|---:|---:|---|
| 1 | 14.54 t/s | 18.22 t/s | 1.253 | 8 slots, usable 12.8 GiB |
| 2 | 8.08 t/s | 11.24 t/s | 1.391 | 8 slots, usable 12.6 GiB |
| 3 | 8.69 t/s | 10.88 t/s | 1.252 | 8 slots, usable 11.2 GiB |
| **median** | — | — | **1.253** | — |

3/3 faster; all first transcripts identical and `state: independent`.

Single auto runs:

| KV | auto slots | load | aggregate | wall |
|---|---:|---:|---:|---:|
| f16 | 8 | 746.7 ms | 20.17 t/s | 12.30 s |
| f32 | 8 | 1846.1 ms | 18.12 t/s | 13.69 s |

f32 auto still fits under the budget but pays a 2.5x load; f16 is the recommended auto pairing.

### Fidelity

| KV | mean KL | top-1 |
|---|---:|---:|
| f32 | 1.130e-11 | 63/63 |
| f16 | 2.501e-07 | 63/63 |
| i8 | 2.373e-04 | 61/63 |

Exactly the previously recorded tiers; the spike changes only the slot-count CLI path.

## 4. Caveats

- **Budget uses total usable, not live free memory.** `MemoryBudget::snapshot().usable` is derived from installed/available RAM and the OS reserve; it said 11–13 GiB usable while the box was under load. Under page-cache pressure, auto can still pick 8 and hit the cycle-11 loss. The flag is opt-in, not the default, for that reason.
- **Floor semantics**: the code prefers 4 but lowers below 4 when the budget cannot fit 4, with a warning. Forcing a hard 4-floor would risk overcommit; the report keeps the safe behavior.
- **Auto is only on `bench continuous`.** `generate`/sessions still take explicit slot counts; a future server layer can reuse the same helper.

## 5. Reimplementation recipe (exact)

### 5.1 Parser (`apps/cli/main.cpp`, `bench continuous`)

```text
else if (flag == "--sequences") {
  if (value == "auto") sequences = 0;       // sentinel
  else ok = parse_int(value, sequences);
}
```

### 5.2 Resolution (`command_continuous_bench`, before validation/load)

```text
if (sequences <= 0) {
  plan_context = context > 0 ? context : 4096;
  file = gguf::File::open(path, false);
  plan = models::analyse(file, plan_context);
  budget = memory::MemoryBudget{};
  decision = models::place(plan, budget.snapshot().usable, plan.context, 8);
  sequences = clamp(decision.max_sessions, 1, 8);
  if (sequences < 4) warn("budget fits only N slots");
  print("auto : N slots (usable X GiB, KV/session Y MiB)");
}
```

Then the existing `Qwen2::load(..., sequences)` path is unchanged.

### 5.3 Gate

- build + 145/145 tests;
- 3 paired `auto` vs fixed 4 at 8×32, ≥1.15x and identical first transcripts;
- f32/f16/i8 fidelity tiers unchanged;
- `python aikit.py check aegean-tge` 4/4.

## 6. Verdict and disposition

**KEEP as an opt-in.** Auto selects 8 when the budget supports it and delivers the measured +25% median aggregate over fixed 4, with exact transcripts and unchanged fidelity. It should not become the default until the policy uses live free-memory/page-cache headroom rather than total usable bytes; the cycle-11 pressure loss is the evidence. The code recipe is complete and small enough to port directly.

## 7. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
tge bench continuous --model Llama-3B --requests 8 --tokens 32 --threads 6 --sequences auto --kv f16
tge bench continuous ... --sequences 4 --kv f16
tge fidelity --model Llama-3B --corpus docs/architecture.md --context 512 --tokens 64 --kv f32|f16|i8
git checkout -- apps/cli/main.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
