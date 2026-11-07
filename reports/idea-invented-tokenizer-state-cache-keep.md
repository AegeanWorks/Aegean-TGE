# INVENTED: Serialized flat tokenizer state — KEEP (warm tokenizer 107.6 → 19.2 ms, 5.6x)

- **Idea**: Combine the two near-miss tokenizer cycles into the mechanism they pointed at: build the tokenizer once with the flat `StringIndex`, serialize the *complete final state* (pieces, byte tables, vocab/merge arenas and slots, control tables, scores, flags) to a keyed sidecar, and restore it directly on later invocations. Warm loads skip both the GGUF metadata parse and the map construction. New invented direction; no prior report serializes the final tokenizer state.
- **Cycle date**: 2026-11-07
- **Verdict**: **KEEP-GRADE.** Warm tokenizer construction: Llama-3B 107.6 → **19.2 ms** median (5.6x, 10/10 pairs), Qwen-0.5B 91.7 → **15.9 ms** (5.8x). Cache-aware `auto` startup: 333.4 → **218.8 ms** median (1.52x). All prompt/generated ids identical, including control tokens; 145/145 tests; 4/4 project gates.
- **Tree state at close**: spike reverted from `include/tge/models/tokenizer.hpp`, `src/models/tokenizer.cpp`, `apps/cli/main.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; the 24.6 MB cache and saved binaries live outside the tree; pre-existing stash untouched; nothing pushed.

## 1. Mechanism

The tokenizer build is ~100–125 ms on Llama-3B. The previous cycles established:

- caching the raw GGUF tokenizer arrays saves only the ~20–25 ms parse (`idea-invented-tokenizer-source-cache-killed.md`);
- replacing the node maps with a flat index saves only the ~20–25 ms node build (`idea-invented-flat-tokenizer-index-killed.md`).

This cycle removes the **whole rebuild**: it serializes the final `Tokenizer` object and restores it.

### 1.1 Flat index

`StringIndex` is the cycle-4 arena-backed open-addressing map:

```text
arena_: one string with every key back-to-back
slots_: {hash u64, offset u32, length u32, value int, state u8}
state:  0 empty, 1 occupied, 2 tombstone
```

It preserves `unordered_map::emplace` first-value semantics, tombstones on erase, and a 0.7-load-factor rehash. Both `vocab_` and `merge_rank_` use it.

### 1.2 State sidecar

After the first successful parse, `save_cache` writes:

| field | form |
|---|---|
| pieces | count + length-prefixed strings |
| byte alphabet | `int32[512]` + `uint32[256]` raw |
| `vocab_` | arena blob + slot array + `size_` + `tombstones_` |
| `merge_rank_` | same |
| `control_` | count + bytes |
| `control_tokens_` | count + (string, id) pairs |
| `control_by_first_` | count + (char, vector<int>) |
| flags | `llama_bpe_digits_`, `spm_` |
| `scores_` | count + `float` array |
| `byte_token_` | `int[256]` raw |

The header is `TGTS` magic (`0x53544754`), version 1, and a key derived from the model path, file size and last-write-time. Any read failure, size mismatch, bad slot state, out-of-range offset/id or key mismatch is a silent miss; the caller rebuilds and rewrites.

`from_path`:

```text
if TGE_TOKENIZER_STATE_CACHE is set:
    locate %TEMP%/tge-tokenizer-state-cache/<key>.bin
    if load_cache succeeds -> return the restored Tokenizer
open the GGUF with full metadata
tokenizer = from_gguf(file)
save_cache(tokenizer)
return tokenizer
```

The CLI `auto` policy now prefers the lazy path when the state cache is enabled, so the model load skips the large tokenizer arrays and the tokenizer comes from the sidecar:

```text
eager = vocabulary == Eager ||
        (vocabulary == Auto && needs_tokenizer && !state_cache)
```

Explicit `--vocabulary eager` still forces the old path; the cache is opt-in via the env flag for this spike.

## 2. Predicted gain, with the math

The two component measurements:

```text
baseline tokenizer             ≈ 107–133 ms
source sidecar alone           ≈ 107 ms (saves parse)
flat index alone               ≈ 101 ms (saves node build)
sidecar + flat build           ≈ 101 - 24 ≈ 77 ms (not enough)
```

Serializing the final tables removes the remaining build, so the warm cost is bounded by:

```text
read 24.6 MB from page cache     ≈ 3–5 ms
deserialize pieces + blobs       ≈ 5–8 ms
resize/assign Vec/arrays         ≈ 5–10 ms
total warm tokenizer             ≈ 15–25 ms
```

Predicted 4–7x on the tokenizer phase. The falsifier was set at **≥3x warm tokenizer**, ids identical, first-run creation ≤50 ms over baseline.

## 3. Falsifier

- **KEEP** only if all hold:
  1. warm tokenizer construction is at least **3x** faster than baseline on interleaved pairs;
  2. prompt ids and generated ids are identical on the thinking set, CJK and control-token prompts;
  3. the full suite is green and first-run cache creation adds no more than ~50 ms.
- **KILL** on any id divergence, crash, malformed-cache acceptance, or a median below 3x.

All passed.

## 4. Measured numbers

### 4.1 Llama-3.2-3B-Q4_K_M, 10 interleaved pairs

`--vocabulary lazy`, 5-token prompt, 6 threads, warm sidecar (24,640,767 bytes).

| pair | baseline tokenizer | state cache | ratio |
|---:|---:|---:|---:|
| 1 | 120.3 | 18.7 | 6.43 |
| 2 | 103.5 | 19.2 | 5.39 |
| 3 | 112.7 | 21.4 | 5.27 |
| 4 | 105.1 | 19.1 | 5.50 |
| 5 | 97.5 | 18.3 | 5.33 |
| 6 | 113.7 | 19.6 | 5.80 |
| 7 | 109.9 | 19.3 | 5.69 |
| 8 | 106.6 | 19.2 | 5.55 |
| 9 | 107.6 | 22.3 | 4.83 |
| 10 | 125.8 | 18.9 | 6.66 |
| **median** | **107.6** | **19.2** | **5.6x** |
| **best** | **97.5** | **18.3** | **5.33x** |

First run with cache creation: tokenizer **110.2 ms**, within the baseline band; the 24.6 MB write is page-cache cheap. Warm runs 18.6–22.3 ms.

### 4.2 Qwen2.5-0.5B-Q8_0 cross-model

| | tokenizer |
|---|---:|
| baseline lazy | 91.7 ms |
| cache creation | 77.9 ms |
| warm state cache | **15.9 ms** (5.8x) |

Prompt ids identical baseline vs first vs second invocation.

### 4.3 Cache-aware `auto` total startup (Llama-3B, 8 interleaved pairs)

`auto` on the spike uses the lazy load + state cache; baseline `auto` uses eager load + rebuild.

| | baseline auto | state-cache auto | ratio |
|---|---:|---:|---:|
| median load + tokenizer | 333.4 ms | **218.8 ms** | **1.52x** |
| best | 274.1 ms | **190.4 ms** | 1.44x |
| median breakdown | 234 ms load + 99 ms tokenizer | 209 ms load + 26 ms tokenizer | — |

All 8 pairs faster. The remaining load (~209 ms) is now dominated by the model's KV/mmap path, not the tokenizer.

## 5. Correctness evidence

| check | result |
|---|---|
| thinking prompt A (13 prompt ids) | promptids + generated ids **IDENTICAL** |
| thinking prompt B (8 prompt ids) | promptids + generated ids **IDENTICAL** |
| CJK prompt (96 prompt ids) | promptids + generated ids **IDENTICAL** |
| Llama control prompt `<|start_header_id|>user<|end_header_id|>Hello` | promptids `128006 882 128007 9906` **IDENTICAL** |
| Qwen 0.5B prompt | promptids identical; first and second invocation identical |
| test suite | **145/145** on spike and after revert |
| project gate | **4/4** after revert (test 17.7 s, hygiene, guidelines, secrets) |

The cache restores `control_`, `control_tokens_` and `control_by_first_`, so literal control-token matching survives; the control prompt above proves it on the real Llama vocabulary. The SPM fields (`scores_`, `spm_`, `byte_token_`) are serialized but were not exercised on a real Phi model in this cycle; the synthetic SPM fixtures still pass through `from_gguf`.

## 6. Fidelity and gates

- **Numerics**: no model arithmetic touched; no fidelity owed.
- **Kernel bench**: no kernel touched; no kernel A/B owed.
- **Build**: `cmake --build --preset release` green (54/54).
- **Tests**: 145/145 before and after; no test changes shipped.
- **Project gate**: `python aikit.py check aegean-tge` → 4/4.

## 7. Reimplementation recipe (exact)

### 7.1 Add `StringIndex` to `tokenizer.hpp`

As in `idea-invented-flat-tokenizer-index-killed.md` §7.1, plus `friend class Tokenizer;` so the cache methods can serialize the private arena/slots. Replace `unordered_map<string,int> vocab_, merge_rank_` with `StringIndex`.

### 7.2 Cache declarations on `Tokenizer`

```text
static bool load_cache(const string& path, uint64 key,
                       unique_ptr<Tokenizer>& tokenizer);
static void save_cache(const string& path, uint64 key, const Tokenizer&);
static bool cache_location(const string& model_path, uint64& key, string& path);
```

### 7.3 Binary format (version 1)

```text
u32 magic 0x53544754 ("TGTS"); u32 version 1; u64 key
u32 piece_count; [u32 len + bytes] * piece_count
byte_of_codepoint: int32[512] raw
codepoint_of_byte: uint32[256] raw
for each of vocab_, merge_rank_:
    u32 arena_len + bytes
    u32 slot_count + raw Slot[slot_count]
    u64 size_; u64 tombstones_
u32 control_count + bytes
u32 control_token_count; [u32 len + bytes, u32 id] *
u32 by_first_count; [u8 char, u32 count, u32 id *] *
u32 llama_bpe_digits; u32 spm
u32 score_count + float[score_count] raw
byte_token: int[256] raw
```

`Slot = {u64 hash, u32 offset, u32 length, int value, u8 state}`; serialize the vector raw. Bump `kStateVersion` whenever the layout or field order changes.

### 7.4 Validation on load

- magic/version/key match;
- piece count 1..2M; each blob ≤64 MB;
- slot count 0 or a power of two, ≤4M;
- every state ≤2; occupied `offset + length ≤ arena.size()`; occupied count == `size_`;
- `size_`, `tombstones_` ≤ slot count;
- control count ≤ pieces; control ids < pieces; by-first ids < pieces;
- score count ≤2M; flags 0/1.

Any failure returns false; `from_path` then rebuilds and overwrites. The cache is disposable.

### 7.5 Cache location and key

```text
key = FNV1a(model_path)
      ^ (file_size * 0x9E3779B97F4A7C15)
      ^ (last_write_time * 0xC2B2AE3D27D4EB4F)
path = temp_directory_path()/tge-tokenizer-state-cache/<key:016x>.bin
```

Enable with `TGE_TOKENIZER_STATE_CACHE=1` for the spike; the shipped form should be a `--tokenizer-cache` flag plus an optional cache directory, with the same key.

### 7.6 `auto` policy

When the cache is enabled and a text prompt needs a tokenizer, choose the lazy model load (skip the large arrays) and let `from_path` restore the state. Keep explicit `--vocabulary eager` as the escape hatch. Do not enable the cache on a read-only / shared temp with no write permission; the miss path simply rebuilds.

### 7.7 Gate

1. build + 145/145 tests;
2. first invocation creates the cache; second invocation warm;
3. interleaved baseline vs cached tokenizer timing ≥3x;
4. thinking + CJK + control prompts id-identical;
5. `python aikit.py check aegean-tge` 4/4.

## 8. Verdict and disposition

**KEEP-GRADE.** The combined mechanism removes the entire tokenizer rebuild rather than shaving it: 5.6x on Llama-3B (107.6 → 19.2 ms, 10/10 pairs), 5.8x on Qwen-0.5B, 1.52x on cache-aware total startup, with exact ids including control tokens and no model arithmetic touched. It is the first clear tokenizer win after the two near-miss kills and supersedes both for shipping purposes.

**Caveats to carry into the port:** the key is size+mtime only (a content change that preserves both would falsely hit); the raw `Slot` layout is build-specific (bump the version on any layout change); the sidecar is 24.6 MB per model in temp and should be evictable; the SPM cache path needs one real Phi run before claiming cross-family coverage.

## 9. Exact commands run (selection)

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_TOKENIZER_STATE_CACHE=1 tge generate --model Llama-3B \
  --prompt "The capital of France is" --n 1 --threads 6 --vocabulary lazy
                                                       → first 110.2 ms, warm 18.6–22.3 ms
<baseline> generate ... --vocabulary lazy               → 97.5–125.8 ms
tge generate --model Qwen-0.5B ... --vocabulary lazy    → 91.7 → 15.9 ms warm
tge generate --model Llama-3B --prompt <thinking|control> --print-ids --text --gpu off
                                                       → ids IDENTICAL
git checkout -- include/tge/models/tokenizer.hpp src/models/tokenizer.cpp apps/cli/main.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
