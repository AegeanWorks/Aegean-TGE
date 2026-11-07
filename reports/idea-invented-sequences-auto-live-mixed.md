# INVENTED: Live-headroom `--sequences auto` — MIXED (safety fixed; 8-slot throughput not reliable)

- **Idea**: Refine the budget-aware `--sequences auto` policy to use **live available physical memory minus a page-cache reserve**, instead of the installed-RAM `usable` budget. Resolve slots from full-context KV headroom, not `place().max_sessions`.
- **Cycle date**: 2026-11-07
- **Verdict**: **MIXED — keep as a safety policy, not as a throughput guarantee.** The live policy behaves correctly: normal conditions pick 8 slots, simulated pressure picks 2 and warns, transcripts identical. It also exposed a real trap in `place()` (it can return `max_sessions=8` by shrinking context to ~75 tokens). But the 8-slot throughput A/B failed the 1.15x bar: 0.704, 1.118, 1.104 (median **1.104x**) despite 9 GiB headroom, so the win is load-dependent. Auto stays opt-in.
- **Tree state**: spike reverted from `apps/cli/main.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; nothing pushed.

## 1. Mechanism

The cycle-12 auto path used `MemoryBudget.snapshot().usable`, which is based on installed RAM minus the OS reserve and current available RAM at construction time. It could choose 8 while page cache/other processes were under pressure. It also used `place().max_sessions`, whose contract shrinks context before sessions: with 2.4 GiB headroom and 8 requested sessions it returned `max_sessions=8` by reducing context to ~75 tokens — the wrong answer for a fixed-context continuous bench.

The refined path:

```text
snapshot = MemoryBudget{}.snapshot()
reserve  = max(total_physical / 16, 1 GiB)       // page-cache/OS headroom
           (env TGE_AUTO_RESERVE_MB overrides for testing)
live     = max(0, available_physical - reserve)
usable   = min(snapshot.usable, live)
headroom = usable - plan.parameter_bytes
slots    = clamp(headroom / plan.kv_bytes_per_session, 1, 8)
```

- `plan` is the existing metadata-only `ModelPlan`; `kv_bytes_per_session` is full-context KV.
- If `headroom <= 0`, auto fails with a clear error instead of overcommitting.
- `slots < 4` prints a warning but uses the safe count; a hard 4-floor would overcommit.
- `auto` remains opt-in; the default stays 4.

## 2. Falsifier

- **KEEP as a throughput preset** if auto ≥1.15x fixed 4 across paired runs, transcripts identical, slots ≤8.
- **Do not default** if it loses or is below the bar; keep the live policy as a safety improvement.

Measured: below the bar (median 1.104), so opt-in only.

## 3. Measured numbers

Llama-3.2-3B-Q4_K_M, `bench continuous --requests 8 --tokens 32 --threads 6 --kv f16`, context 4096, 248 decode tokens.

### Normal vs simulated pressure

| case | slots | live | usable | headroom | load | aggregate | first |
|---|---:|---:|---:|---:|---:|---:|---|
| normal | 8 | 12.7 GiB | 10.7 GiB | 8.8 GiB | 745.7 ms | 12.37 t/s | reference |
| reserve 9000 MB | **2** | 13.0 GiB | 4.2 GiB | 2.4 GiB | 199.9 ms | 12.99 t/s | IDENTICAL |

The pressure case prints “below the preferred floor of 4” and runs 2 slots rather than overcommitting. `TGE_AUTO_RESERVE_MB=9000` is a test override for the page-cache reserve; the production policy uses the 1 GiB/6.25% constant.

### Auto (8 slots) vs fixed 4, 3 interleaved pairs

| pair | fixed 4 | auto | ratio | first |
|---:|---:|---:|---:|---|
| 1 | 19.22 t/s | 13.54 t/s | 0.704 | IDENTICAL |
| 2 | 17.35 t/s | 19.40 t/s | 1.118 | IDENTICAL |
| 3 | 17.84 t/s | 19.69 t/s | 1.104 | IDENTICAL |
| **median** | — | — | **1.104** | — |

2/3 faster, but one 30% loss; median below the 1.15x bar. Auto picked 8 in all three runs with 9.2–9.5 GiB headroom, so the loss is not a memory-budget miss — it is CPU/box contention. Live memory is necessary but not sufficient for a throughput decision.

### Fidelity (unchanged)

| KV | mean KL | top-1 |
|---|---:|---:|
| f32 | 1.130e-11 | 63/63 |
| f16 | 2.501e-07 | 63/63 |
| i8 | 2.373e-04 | 61/63 |

The spike changes only CLI slot selection; model arithmetic is untouched.

## 4. Findings

1. **`place()` is not a slot-count oracle.** Its `max_sessions` may preserve the session count by shrinking context; auto must divide full-context KV headroom directly. This was a live bug in the cycle-12 policy and is the main correctness result of the cycle.
2. **Live memory fixes overcommit, not throughput.** The pressure fallback works, but 8 slots can still lose on a loaded box even with 9 GiB headroom; CPU contention/thermal state is invisible to the memory snapshot.
3. **The 4-floor is a preference, not a guarantee.** If the live budget fits only 2, auto runs 2 and warns; forcing 4 would be an overcommit.
4. **Opt-in remains right.** Cycle 11 saw a 1.337x median, cycle 12 1.253x, this cycle 1.104x with one loss. The 8-slot gain is real but not robust enough for a default.

## 5. Reimplementation recipe (exact)

### 5.1 Parser

```text
else if (flag == "--sequences") {
  if (value == "auto") sequences = 0;
  else ok = parse_int(value, sequences);
}
```

### 5.2 Resolution (before load)

```text
plan_context = context > 0 ? context : 4096
file = gguf::File::open(path, false)
plan = models::analyse(file, plan_context)
snapshot = memory::MemoryBudget{}.snapshot()
reserve = max(snapshot.total_physical / 16, 1 GiB)   // env override for tests
live = max(0, snapshot.available_physical - reserve)
usable = min(snapshot.usable, live)
headroom = usable - plan.parameter_bytes
if headroom <= 0 or plan.kv_bytes_per_session <= 0: fail
slots = clamp(headroom / plan.kv_bytes_per_session, 1, 8)
if slots < 4: warn
printf("auto: %d slots (live ..., usable ..., headroom ..., KV/session ...)")
```

Then pass `slots` to the existing `Qwen2::load(..., sequences)`.

### 5.3 Gate

- build + 145/145 tests;
- normal auto picks 8; `TGE_AUTO_RESERVE_MB=9000` picks 2 and warns;
- 3 paired auto-vs-4 runs, transcripts identical;
- f32/f16/i8 fidelity tiers unchanged;
- `python aikit.py check aegean-tge` 4/4.

## 6. Verdict and disposition

**MIXED — keep as an opt-in safety policy.** The live-headroom calculation is correct and prevents the `place()` context-shrink overcommit; the simulated pressure case proves the fallback. But the 8-slot throughput win is not reliable on a loaded box (median 1.104x here vs 1.253x/1.337x earlier), so it must not become the default. A future policy needs both live memory and a CPU-load/throughput signal (or a short calibration) before choosing 8; until then, `--sequences auto` is a safer opt-in, not a performance promise.

## 7. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
tge bench continuous --model Llama-3B --requests 8 --tokens 32 --threads 6 --sequences auto --kv f16
TGE_AUTO_RESERVE_MB=9000 tge bench continuous ... --sequences auto --kv f16
tge bench continuous ... --sequences 4 --kv f16
tge fidelity ... --kv f32|f16|i8
git checkout -- apps/cli/main.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
