# INVENTED: CPU-load guard for `--sequences auto` — KILLED (8 slots still win under synthetic CPU load)

- **Idea**: Add a CPU-busy signal to the live-headroom `--sequences auto` policy: sample `GetSystemTimes` for 80 ms before load; if busy >70%, cap the slot count at 4, otherwise allow 8. The premise: the cycle-11 8-slot loss was CPU contention, so a busy box should stay at 4.
- **Cycle date**: 2026-11-07
- **Verdict**: **KILLED by measurement.** Under four synthetic CPU hogs the guard capped at 4 slots and delivered 12.45 t/s, while the same auto policy without the guard (8 slots) delivered **15.96 t/s**; fixed 4 gave 14.58 t/s. Even at “100% busy,” 8 slots won. CPU busy is not a proxy for the 8-slot throughput trade; the guard removes the win it was meant to protect.
- **Tree state**: spike reverted from `apps/cli/main.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; nothing pushed.

## 1. Mechanism

On top of the cycle-13 live-memory policy:

```text
reserve  = max(total/16, 1 GiB)
live     = max(0, available_physical - reserve)
usable   = min(snapshot.usable, live)
headroom = usable - parameter_bytes
slots    = clamp(headroom / kv_per_session, 1, 8)
busy     = 1 - idle_delta/total_delta over 80 ms   (GetSystemTimes)
if busy > 0.70: slots = min(slots, 4)
```

The 80 ms sample runs before the model load; `TGE_AUTO_CPU=0` disables it for the A/B.

## 2. Falsifier

- **KEEP** only if, under synthetic CPU load, the guard picks 4 and is not slower than fixed 4, and the 8-slot path is not better under the same load.
- **KILL** if 8 slots still win under CPU load; then CPU busy is the wrong signal.

Measured: 8 wins; kill.

## 3. Measured numbers

Llama-3.2-3B-Q4_K_M, `bench continuous --requests 8 --tokens 32 --threads 6 --kv f16`, 248 decode tokens.

### No synthetic hogs

| policy | slots | CPU sample | aggregate |
|---|---:|---:|---:|
| guard | 4 | 100% | 16.85 t/s |
| no guard | 8 | 0% (sample disabled) | 17.90 t/s |

The box already reports 100% busy from other processes, so the guard caps at 4 almost always. Even then, 8 slots was 6% faster.

### Four synthetic CPU hogs

| policy | slots | CPU sample | aggregate |
|---|---:|---:|---:|
| guard | 4 | 100% | 12.45 t/s |
| no guard | 8 | 0% (sample disabled) | **15.96 t/s** |
| fixed 4 | 4 | — | 14.58 t/s |

8 slots is **28% faster than the guarded 4** and 9.5% faster than fixed 4 under the same hogs. All first transcripts identical and `state: independent`.

The guard-vs-fixed-4 15% gap at the same slot count is box noise (the guard adds only the 80 ms sample, not decode work); the decisive comparison is guard (4) vs no-guard (8), which is outside the noise band and opposite to the guard’s premise.

### Fidelity (unchanged)

| KV | mean KL | top-1 |
|---|---:|---:|
| f32 | 1.130e-11 | 63/63 |
| f16 | 2.501e-07 | 63/63 |
| i8 | 2.373e-04 | 61/63 |

The spike changes only CLI slot selection; model arithmetic is untouched.

## 4. Why it failed

- **CPU busy is not slot efficiency.** The 8-slot win comes from sharing one weight stream across eight decodes; CPU contention slows both 4 and 8, but 8 still amortizes the weights better.
- **The sample is almost always 100% on this box** even without our hogs (VS Code/Defender/background), so the guard is not a discriminator; it just disables 8 slots.
- **The cycle-11 loss was likely noise/memory, not CPU.** Cycle 11 saw 1.337x median, cycle 12 1.253x, and this cycle’s no-guard 8 won under both normal and synthetic load. A single loss is not enough to justify a CPU cap.
- **A throughput calibration is the right signal, not system CPU**: run a short 4-vs-8 active-slot A/B after load and pick the winner; that directly measures the trade on the current box. It needs a factored mini-run, which this cycle did not build.

## 5. Reimplementation recipe (kill-grade, for the record)

1. Add `sample_cpu_busy(ms)` in `apps/cli/main.cpp` (`_WIN32` `GetSystemTimes`; busy = 1 − idle/total; fallback 0).
2. In the `--sequences auto` block, after the live-headroom slot count:
   `if (getenv("TGE_AUTO_CPU") != "0") { busy = sample_cpu_busy(80); if (busy > 0.70) slots = min(slots, 4); }`
3. Print the sample in the `auto:` line.
4. Gate: normal + four `Start-Job { while($true){} }` hogs; compare guard vs `TGE_AUTO_CPU=0` vs fixed 4.
5. Expected result from this cycle: guard 4 = 12.45 t/s, no-guard 8 = 15.96 t/s → do not ship the guard.

## 6. Verdict and disposition

**KILLED.** Remove the CPU-busy cap. Keep the live-memory-only auto from cycle 13 as an opt-in; it is a safety improvement, not a throughput guarantee. The next attempt should be a short **throughput calibration** (4 vs 8 active slots) if the auto policy is to choose 8 reliably; system CPU is not a usable proxy.

## 7. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
tge bench continuous --model Llama-3B --requests 8 --tokens 32 --threads 6 --sequences auto --kv f16
TGE_AUTO_CPU=0 tge bench continuous ... --sequences auto --kv f16
# with four CPU hogs:
1..4 | ForEach-Object { Start-Job { while ($true) { } } }
tge bench continuous ... --sequences auto --kv f16
tge bench continuous ... --sequences 4 --kv f16
Stop-Job/Remove-Job
tge fidelity ... --kv f32|f16|i8
git checkout -- apps/cli/main.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
