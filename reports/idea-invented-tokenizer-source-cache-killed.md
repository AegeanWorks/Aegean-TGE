# INVENTED: Tokenizer source sidecar — KILLED (quiet-box 1.24x, below the 1.25x bar; map build dominates)

- **Idea**: Cache the tokenizer-relevant GGUF arrays (tokens, merges, token types, scores, splitter name, model name) in a keyed binary sidecar, so warm text-prompt invocations skip reparsing the full model metadata and only rebuild the tokenizer maps. New invented direction; the measurement log explicitly lists a cached vocabulary sidecar as an open item (§8/§13.3), and no prior report implements or prices it.
- **Cycle date**: 2026-11-07
- **Verdict**: **KILLED on the stated falsifier.** Quiet-box paired median 1.243x (best-of-best 1.238x) against a pre-registered 1.25x bar. The sidecar is functionally correct and id-identical, but the saved GGUF parse (~24 ms) is smaller than predicted, and the ~65–80 ms `unordered_map` construction remains the floor. Do not ship the cache alone; reopen only with a serialized flat hash table.
- **Tree state at close**: spike reverted from `include/tge/models/tokenizer.hpp` and `src/models/tokenizer.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; temp cache and saved spike binary are outside the tree; pre-existing stash untouched; nothing pushed.

## 1. Mechanism

For a text prompt, the CLI builds the tokenizer through one of two paths:

- **eager**: the model loader parses all GGUF metadata, then `Tokenizer::from_gguf` builds the maps;
- **lazy**: the model loader skips large arrays, then `Tokenizer::from_path` opens the GGUF a second time with full metadata, extracts the tokenizer arrays, and builds the maps.

`Tokenizer::from_path` therefore pays two costs every invocation: GGUF metadata parse (copying ~400k strings across tokens + merges) and tokenizer construction (`pieces_`, `vocab_`, `merge_rank_`, control tables).

The spike factored `from_gguf` into:

```text
read_source(file) -> Source { tokens, merges, types, scores, pre, model }
from_source(Source) -> Tokenizer
```

`from_path` then:

1. computes a cache key from the model path, file size and last-write-time;
2. tries to load `Source` from `%TEMP%\tge-tokenizer-cache\<key>.bin`;
3. on a hit, calls `from_source` directly (no GGUF open);
4. on a miss, opens the GGUF, reads `Source`, builds, and writes the sidecar.

The sidecar stores raw arrays, not the maps: tokens/merges as length-prefixed strings, types as `u32`, scores as `f32`, plus the splitter/model names. It is versioned (`TGTK`, v1), keyed, bounds-limited (2M entries, 16 MiB/string), and any read failure is a silent miss.

## 2. Predicted gain, with the math

Measured before the spike:

| component | estimate |
|---|---:|
| lazy tokenizer total | ~103 ms |
| GGUF metadata parse (eager vs skipped load delta) | ~38 ms |
| map + piece construction | ~65 ms |

If the sidecar removes the parse and adds a small binary read:

```text
warm tokenizer ≈ 5–10 ms (read Source) + 65 ms (build maps) ≈ 70–75 ms
speedup ≈ 103 / 72 ≈ 1.43x
```

The falsifier was set at **≥1.25x** to leave room for the sidecar read and box noise.

## 3. Falsifier

- **KEEP** only if all hold:
  1. warm sidecar tokenizer construction is at least **1.25x** faster than baseline on the paired quiet-box runs;
  2. prompt ids and generated ids are identical on the fixed fixtures and thinking set;
  3. first-run cache creation costs no more than 50 ms over baseline.
- **KILL** on any id divergence, crash, or a median below 1.25x.

Result: ids and creation passed, speed failed at 1.243x.

## 4. Baseline reproduction (pre-spike tree)

`--vocabulary lazy`, Llama-3.2-3B-Q4_K_M, 5-token prompt, 6 threads, `--n 1`:

- baseline lazy tokenizer: **107.0–159.7 ms** across the paired runs (median 133.5 ms);
- first spike run (creates the 6,183,682-byte cache): **133.8 ms**;
- warm spike tokenizer: **86.4–151.7 ms** (median 107.4 ms).

The clean pre-spike estimate of a 38 ms parse was too high for this file; the actual saving is ~24–26 ms.

## 5. Paired A/B (quiet box, 10 alternating launches)

Both binaries: same release build, same model, same prompt, `--vocabulary lazy`, `--n 1`, `--threads 6`. Baseline and spike launched back-to-back per pair; the spike has `TGE_TOKENIZER_CACHE=1` and the warm sidecar.

| pair | baseline tokenizer ms | sidecar tokenizer ms | ratio |
|---:|---:|---:|---:|
| 1 | 159.7 | 151.7 | 1.053 |
| 2 | 155.1 | 123.3 | 1.258 |
| 3 | 149.7 | 112.9 | 1.326 |
| 4 | 124.8 | 101.8 | 1.226 |
| 5 | 141.2 | 98.1 | 1.439 |
| 6 | 124.2 | 124.8 | 0.995 |
| 7 | 136.1 | 117.5 | 1.158 |
| 8 | 130.8 | 86.4 | 1.514 |
| 9 | 116.9 | 91.5 | 1.278 |
| 10 | 107.0 | 97.6 | 1.096 |
| **median** | **133.5** | **107.4** | **1.243** |
| **best** | **107.0** | **86.4** | **1.238** |

A second 8-pair set was taken after the box became heavily loaded (baseline 123–287 ms, spike 95–187 ms). Its median ratio is ~1.49, but the load inflates the baseline more than the cache path and it is **not used for the verdict**. The quiet set is the gate.

### Why it missed

The sidecar still reads 6.18 MB and reconstructs ~400k `std::string`s before the maps; that costs ~10–15 ms. The GGUF parse it replaces is ~20–25 ms, not 38 ms. Net saving ~24 ms, while the map build is ~65–80 ms. The ceiling of this approach on this file is therefore ~1.25–1.3x, and the box noise is of the same size as the margin.

## 6. Behavioural parity

| path | comparison | result |
|---|---|---|
| thinking prompt A (13 prompt ids) | promptids + generated ids vs baseline | IDENTICAL |
| thinking prompt B (8 prompt ids) | promptids + generated ids vs baseline | IDENTICAL |
| CJK prompt (96 prompt ids) | promptids + generated ids vs baseline | IDENTICAL |
| SPM synthetic fixtures | 145-test suite (`from_gguf` → `read_source` → `from_source`) | PASS |
| control/user-defined fixtures | 145-test suite | PASS |

The refactor itself is correct; the cache changes no ids.

## 7. Fidelity and gates

- **Numerics**: none touched; no fidelity owed.
- **Kernel bench**: none touched; no kernel A/B owed.
- **Build**: `cmake --build --preset release` green (54/54).
- **Tests**: 145/145 before and after the spike; the baseline was rebuilt after revert.
- **Project gate**: `python aikit.py check aegean-tge` → **4/4 gates passed** (test 20.1 s, hygiene, guidelines, secrets) after revert.

## 8. Reimplementation recipe (exact, for the record — kill-grade alone)

### 8.1 `include/tge/models/tokenizer.hpp`

Add a private `Source` struct and static helpers:

```text
struct Source {
  vector<string> tokens, merges;
  vector<uint32_t> types;
  vector<float> scores;
  string pre, model;
};
static bool read_source(const gguf::File&, Source&, string& error);
static unique_ptr<Tokenizer> from_source(Source, string& error);
static bool load_source(const string& path, uint64_t key, Source&, string& error);
static void save_source(const string& path, uint64_t key, const Source&);
static bool cache_location(const string& model_path, uint64_t& key, string& path);
```

Change `enable_spm` to take `const vector<float>& scores` instead of `const gguf::File&`.

### 8.2 `src/models/tokenizer.cpp`

1. `read_source`: copy `tokenizer.ggml.tokens`, `.merges`, `.token_type`, `.scores`, `.pre`, `.model` from the `gguf::File` into `Source`; validate the tokens array.
2. `from_source`: the existing `from_gguf` build body, but from `Source`; move `source.tokens` into `pieces_`; build the byte alphabet, `vocab_`, `merge_rank_`, control tables, `llama_bpe_digits_` from `source.pre`, and arm SPM when `source.model == "llama"` and `source.merges.empty()`.
3. `from_gguf` = `read_source` + `from_source(std::move(source))`.
4. Cache key: FNV-1a of the path XOR `size * 0x9E3779B97F4A7C15` XOR `mtime * 0xC2B2AE3D27D4EB4F`; file name `%016llx.bin` under `temp_directory_path()/tge-tokenizer-cache`.
5. Sidecar header: `TGTK` magic (`0x4B544754`), version 1, key `u64`; then token count + length-prefixed strings, merge count + strings, type count + `u32`s, score count + `f32`s, `pre` string, `model` string. Caps: 2,000,000 entries, 16 MiB per string; any read failure is a miss.
6. `from_path`: if `TGE_TOKENIZER_CACHE` is set and not `0`, try `cache_location` + `load_source` + `from_source`; on any miss, parse the GGUF, `save_source` (before moving), then build. On a cache hit that fails to build, clear the error and fall back to the GGUF.
7. Use `resize` on `Source` vectors and read straight into the string data; write the sidecar with `ofstream(binary|trunc)` and remove it on a failed write.

### 8.3 Why not ship

The recipe is complete and safe, but the measured ceiling is below the bar. The next lever is not more cache plumbing: replace `unordered_map<string,int>` with a flat open-addressing string index (stable key arena + offset/length/value slots, tombstones for the special-token erasures), then serialize the final table. That removes the 65–80 ms rebuild, making warm construction ~10–20 ms; the sidecar becomes worth its complexity at that point.

## 9. Verdict and disposition

**KILLED on the stated 1.25x bar.** The mechanism is real and correct — warm construction saves ~24–26 ms and all ids match — but the measured quiet-box median is 1.243x (best 1.238x), and the residual map build dominates. The cache is not worth its invalidation surface on its own. Reopen only with the flat-map/table-serialization follow-up, where the sidecar can skip the rebuild instead of merely shortening the parse.

## 10. Exact commands run (selection)

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145 (spike and baseline)
TGE_TOKENIZER_CACHE=1 tge generate --model Llama-3B --prompt "The capital of France is" \
  --n 1 --threads 6 --vocabulary lazy                  → first run 133.8 ms (creates 6.18 MB cache)
tge generate ... --vocabulary lazy                      → warm baseline 107.0–159.7 ms
TGE_TOKENIZER_CACHE=1 <saved spike> generate ...        → warm sidecar 86.4–151.7 ms
tge generate ... --print-ids --text --gpu off           → thinking/CJK ids IDENTICAL
git checkout -- include/tge/models/tokenizer.hpp src/models/tokenizer.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
