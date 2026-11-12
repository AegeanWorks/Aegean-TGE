# MEASURED: speculation at 2k context on the reference 3B — still 2.2-2.8x with identical ids; un-pinning attention to the grouped schedule is neutral

- **Question**: every speculation measurement in the record is at short prompts (or the 4096-token id sweep on 0.5B-class work). At 2k on the reference **Llama-3.2-3B-Instruct-Q4_K_M** decode attention is 21% of the token and the verifier runs 6 rows per step. (a) Does prompt-lookup speculation still pay at 2k? (b) The shipped tree pins `attention=Head` under `--speculative` for verifier/decode path equivalence, but the verifier's 6-row forward already uses the block/grouped path, so the 1-row correction is the only consumer of the pin — does un-pinning to `gqa` buy the K/V sharing at 2k?
- **Cycle date**: 2026-11-11
- **Verdict**: **MEASURED.** 2,050-token periodic prompt, `--context 2560 --n 32 --kv f16 --threads 10 --gpu off`, greedy:
  - **(a) Spec still wins big at 2k**: 11.50/12.88/10.74 -> **31.56/28.89/30.03 t/s**, ratios **2.744/2.243/2.796** (median **~2.74x**), 6 steps, 5.17 accepted/step, **ids identical** (`4C-1C-DE…`) in all six runs. The attention share grows to 21% of decode (195 ms of 951 ms in the reference run) but the step reduction still dominates.
  - **(b) Un-pinning `head` -> `gqa` under spec is neutral**: 28.91/29.76/30.11 (head) vs 28.81/30.46/29.88 (gqa), ratios 0.997/1.024/0.992, **median ~0.997x**, ids **identical** (`4C-1C-DE…`), accepted/step identical (5.17). The verifier's 6 rows dominate the step; the 1-row correction's attention schedule (1 of 6 rows) cannot move the total. The pin costs nothing measurable and stays.
- **Tree state at close**: spike reverted from `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Llama-3.2-3B-Instruct-Q4_K_M.gguf |
| prompt | fox sentence x205 = 2,050 tokens |
| context | 2560 |
| threads / KV / GPU | 10 / f16 / off |
| n | 32 greedy, `--print-ids` |
| (a) arms | no spec vs `--speculative` |
| (b) arms | `--speculative` (auto->Head pin) vs `--speculative --attention gqa` with a spike env gate `TGE_SPEC_GQA=1` that skips the pin |

Reference run timers (spec, head): `prefill 2050 tokens in 35285.8 ms (58.1 tok/s)`, `decode 31 tokens in 951.3 ms (32.59 tok/s)`, phases `matvec 696.4 ms (74%), attn 195.0 ms (21%), norm 50.3 ms (5%)`, `spec 6 steps, 31 accepted (5.17/step), 36 emitted`.

## 2. Results

### (a) Speculation at 2k

| pair | off t/s | spec t/s | ratio | spec line | ids |
|---:|---:|---:|---:|---|---|
| 1 | 11.50 | 31.56 | 2.744 | 6 steps, 5.17/step | identical |
| 2 | 12.88 | 28.89 | 2.243 | 6 steps, 5.17/step | identical |
| 3 | 10.74 | 30.03 | 2.796 | 6 steps, 5.17/step | identical |
| **median** | 11.50 | 30.03 | **2.744** | | 3/3 |

### (b) Head pin vs grouped under spec

| pair | head t/s | gqa t/s | ratio | ids |
|---:|---:|---:|---:|---|
| 1 | 28.91 | 28.81 | 0.997 | identical |
| 2 | 29.76 | 30.46 | 1.024 | identical |
| 3 | 30.11 | 29.88 | 0.992 | identical |
| **median** | 29.76 | 29.88 | **0.997** | 3/3 |

## 3. Reading

- Speculation is not a short-context trick on this model: at 2k with f16 KV it still delivers ~2.7x on repetitive text with byte-identical greedy ids, even though attention is 21% of the decode token. The step count (6 for 31 tokens) is the lever, and the 6-row verifier's matvec still dominates (74% matvec, 21% attention).
- The per-head pin under speculation is **free**: relaxing it to the grouped schedule neither helps nor hurts (0.997 median) because the verify rows — which already use the grouped block path — dominate; only the rare 1-row correction consults the decode schedule. The pin can stay (equivalence-conservative) with no performance cost, and the measured id identity between the two modes at 2k is a useful equivalence data point on repetitive text (not proof on near-tie prompts).
- Usage guidance: `--speculative` at 2k is worth it on repetitive/templated prompts (2.2-2.8x), matching the 4096-token sweep's 1.95-2.15x trend on weaker hardware; the earlier `--speculative auto` probation remains the right default for unknown prompts.
- No code ships from this cycle; the value is the 2k data point on the reference model and the confirmation that the attention pin is not a hidden tax.

## 4. Reimplementation recipe

1. (a) No code: `tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt "<fox x205>" --context 2560 --n 32 --kv f16 --threads 10 --gpu off --print-ids [--speculative]`, alternating pairs; expect off 10-13 t/s, spec 29-32 t/s, ids identical.
2. (b) Spike the pin: in `command_generate` (`apps/cli/main.cpp:580-587`), gate the `set_attention_mode(Head)` branch behind `std::getenv("TGE_SPEC_GQA") == nullptr`; run the spec arms with and without `TGE_SPEC_GQA=1 --attention gqa`; expect ~1.00x and identical ids. Revert.
3. No fidelity run owed: both arms are exact greedy (`kq8` prefill, float decode), and ids match.

## 5. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge generate ... --prompt "<fox x205>" --context 2560 --n 32 --kv f16 --threads 10 \
  --gpu off --print-ids [--speculative]                            -> §2(a)
tge generate ... --speculative [--attention gqa with TGE_SPEC_GQA=1] -> §2(b)
git checkout -- apps/cli/main.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the rebuilt binary; the committed tree is unchanged except this report.
