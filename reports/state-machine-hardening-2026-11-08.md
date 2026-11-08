# State-machine hardening: implicit states made explicit (2026-11-08)

- **Scope**: `projects/aegean-tge` only. The audit covered the CLI front end, model/generation core, memory/GPU/threading, and tokenizer/GGUF/error contracts.
- **Method**: four parallel code surveys, then a second independent read of every claim that became a proposal. No arithmetic or kernel change is included.
- **Result**: 11 PASS proposals implemented; 6 higher-risk proposals deferred with evidence. Tests 146 -> 158; `aikit.py check aegean-tge` 4/4.

## 1. Implicit state machines found

| Machine | States | The gap that mattered |
|---|---|---|
| CLI dispatch/parse | command -> option loop -> run | three parser families, dropped trailing args, `--help` parsed as a value, errors on stdout |
| KV tier | f32 / f16 / i8 / auto | `f16`/`i8` never cleared `auto`, so an earlier `--kv auto` overrode a later explicit flag; only `generate` cleared it on `f32` |
| Sampling | greedy / temperature / top-k / top-p / min-p / penalty | invalid values threw inside the decode loop after a full model load, aborting the process |
| Speculative decode | plain / greedy / fused / sampled / stopped | silent attention rewrite and silent disable for sampling cuts |
| Attention schedule | auto / scalar / head / gqa, sparse, block prefill | resolved schedule invisible; three position/span conventions |
| KV slots | f32/f16/i8 x N sequences | `forward` addresses slot 0 only; no per-slot valid length |
| Prefix CoW table | empty/populated/hit/miss/evict | process-global, shape-tagged, non-atomic counters |
| Thread pools | idle/published/running/done | two default thread counts; `--threads` meaning undocumented |
| Memory plan | unplanned/placed/clamped | `place(..., sessions=1)` while KV is allocated x`sequences` |
| GPU backend | disabled/unavailable/ready | one-shot unavailable latch; per-call failure invisible |
| ISA dispatch | feature flags -> kernel family | `kq8` default-on called AVX2 code without an AVX2 guard |
| Tokenizer construction | BPE / SPM / llama-bpe / control | family selection silent; docs claimed Viterbi |
| Tokenizer encode/decode | match_special, byte fallback | decode silently drops bad ids; USER_DEFINED printing vs README |
| GGUF parse | open -> metadata -> tensors -> data | unchecked `bytes` multiply and wrapping `data_offset + offset + bytes` |
| Stop set | running/stopped | empty set possible with no warning; Qwen rule was case-sensitive |
| Continuous batch slot | free -> prefilled/decoded -> done | first admitted token not end-checked; prompt length not bounded by context |

## 2. Implemented proposals (PASS)

### P01 - last-wins `--kv` in all four parsers
One `tge::cli::KvSelection` helper; every valid value sets all three booleans. `generate`, `fidelity`, `bench continuous` and `bench sessions` now agree. Before: `--kv auto --kv i8` resolved to auto (f16 on the default context); after: i8. `--kv auto --kv f32` in fidelity used to resolve auto; now f32.

### P02 - validate impossible combinations before load
`validate_preflight` rejects sparse+i8, bad sparse window/stride, missing F16C for `--kv f16`, and missing AVX-VNNI for `--vnni` before `Qwen2::load`. Before: `--sparse --kv i8` paid a load (two loads in fidelity) then failed; after: exit 2 immediately.

### P03 - sample validation before load, and one error boundary
`validate_sampling_params` is now declared in the public header. `generate` calls it after parsing and before loading, so `--top-p 2` exits 2 with a message instead of aborting after prefill. `main` wraps dispatch: `invalid_argument` -> 2, any other `std::exception` -> 1 with a `tge:` line.

### P04 - resolved-state report
Every `generate`/`fidelity` run prints one append-only line:

```text
resolved : kv=f16 (auto) attention=auto decode_threads=10 prefill_threads=16 kernel=kq8 gpu=unavailable speculation=off
```

It exposes the effective KV tier and whether it came from `auto`, the attention schedule, both thread counts, the kernel path, GPU state and the speculation mode (including `disabled (sampling cuts)`).

### P05 - complete `info --json`
The CPU object stays at the top level; `memory`, `budget` (with all four pools), `kernels`, `threads`, `gpu` and `auto_rules` are added objects. No existing key changed.

### P06 - stop-set correctness
The Qwen end-token rule uses the same lowercased architecture as RoPE, so `Qwen2` and `qwen2` agree. A stop id that is present but not an integer is rejected as a malformed contract instead of becoming token 0. `generate` warns when the stop set is empty (stopping only at `--n`).

### P07 - library ISA gates
`kq8_effective()` is `requested && AVX2 && FMA` and the batch dispatcher uses it. `Qwen2::load` throws for `kv f16` without F16C; `set_vnni(true)` throws without AVX-VNNI. The CLI preflight catches both before load; direct API callers get a loud error instead of SIGILL.

### P08 - wrap-safe GGUF tensor checks
The tensor byte size is checked before multiplication, and containment uses a wrap-safe subtraction helper (`tensor_range_fits`). A forged shape whose byte size wraps to 16 is now rejected; before, the parser accepted it as a 16-byte tensor.

### P09 - CLI error hygiene
`--help`/`-h` is recognised for info, gguf, plan, generate, fidelity and every bench subcommand. Errors print usage on stderr, not stdout. Dangling values are named (`bench sessions: '--model' needs a value`), invalid values name the flag, and an unknown bench subcommand is an error instead of falling into `bench gemm`.

### P10 - documentation corrected
`docs/architecture.md` no longer claims "one token at a time, greedy, no tokenizer, no batching, 2048 context" or a 64-token-only prefill; it describes the shipped 128/64 chunk rule, the query-block attention path and both tokenizer families. `tokenizer.hpp` no longer claims Viterbi or table-lookup decoding. README states that CONTROL pieces are omitted and USER_DEFINED pieces print.

### P11 - continuous-batch admission
A newly admitted request whose first token is an end id, or a one-token request, finishes immediately instead of decoding an extra token. `prompt-tokens + tokens` is checked against the model context before the pass.

## 3. Verification

- `scripts\test.cmd`: **158 cases, 0 failed** (146 before; 12 new).
- `python aikit.py check aegean-tge`: **4/4** (test, hygiene, guidelines, secrets).
- Fidelity at 399 positions, Llama-3.2-3B-Q4_K_M: f32 mean KL **1.020e-11**, worst 2.330e-10, top-1 **399/399**; f16 mean **3.252e-07**, top-1 **399/399** — the same values as before the change.
- Manual matrix:

| Command | Before | After |
|---|---|---|
| `generate --kv auto --kv i8` | kv=auto -> f16 | kv=i8 (requested) |
| `fidelity --kv auto --kv f32` | kv=auto -> f16 | kv=f32 (requested) |
| `generate --top-p 2` | abort after load | exit 2 before load |
| `generate --sparse --kv i8` | exit 2 after load | exit 2 before load |
| `info --json` | CPU only | memory/budget/kernels/threads/gpu/auto_rules |
| `bench --rows 8` | "bench gemm: unknown option" | "bench: unknown subcommand '--rows'", exit 2 |
| no arguments | usage on stdout | usage on stderr, exit 2 |
| `generate --help` | "'--help' needs a value" | focused usage, exit 0 |

## 4. Deferred (with reasons)

- **Sequence-aware `forward` + per-slot reset**: no production caller; `forward_batch` covers multi-slot work. Needs an API decision and a multi-slot test model.
- **Plan `sequences` / KV tier, and load-order `--kv auto`**: changes admission and context clamps; needs its own measured task.
- **Prefix-table ownership/thread safety**: redesign, not a patch.
- **GPU per-call failure telemetry and retry**: backend work without an SDK in CI.
- **Tokenizer strict decode / true SPM round-trip**: a product decision on round-trip identity.
- **JSON envelopes for every command**: P05 covers the highest-value one.

## 5. Commands

```text
cmake --preset release; cmake --build --preset release
.\build\release\bin\tge_tests.exe                         -> 158 cases, 0 failed
python aikit.py check aegean-tge                          -> 4/4
.\build\release\bin\tge.exe fidelity --model <Llama-3B> --corpus docs\architecture.md \
  --context 512 --tokens 400 --threads 6 --gpu off --kv f32|f16
```
