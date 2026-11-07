# INVENTED: Flat tokenizer string index — KILLED (1.22x median, under the 1.25x bar; correct but too small alone)

- **Idea**: Replace the tokenizer's `std::unordered_map<std::string,int>` vocab and merge tables with a single-arena, open-addressing `StringIndex`: keys stored in one `std::string`, slots holding arena offsets, first-insert duplicate semantics, tombstone erasure and compaction rehash. This is the lever named by the tokenizer source-cache kill (`idea-invented-tokenizer-source-cache-killed.md`): remove the 65–80 ms node-map construction.
- **Cycle date**: 2026-11-07
- **Verdict**: **KILLED on the 1.25x construction bar.** Two independent interleaved sets measured 1.245x and 1.205x median (combined 22-pair median **1.22x**, best-of-best **1.18x**) — below the bar and far below the 1.4–1.6x prediction. The index is functionally correct (146/146 with a direct contract test, all thinking ids identical), but the map build is not the 65–80 ms the cache cycle assumed; it is ~20–25 ms of a ~100–125 ms construction. The next lever is the combined serialized flat table, not this map alone.
- **Tree state at close**: spike reverted from `include/tge/models/tokenizer.hpp`, `src/models/tokenizer.cpp`, `tests/test_gguf.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; saved baseline/spike binaries are outside the tree; pre-existing stash untouched; nothing pushed.

## 1. Mechanism

The tokenizer builds two string→int maps on every load:

- `vocab_`: piece text → token id (~128k pieces on Llama-3B);
- `merge_rank_`: `"left right"` → merge rank (~280k rules).

The stock `std::unordered_map` allocates a node and a copied key per entry, hashes every key, and chases buckets on lookup. The spike replaced both with `StringIndex`:

```text
arena_: one std::string holding every key back-to-back
slots_: open-addressed table of {hash, offset, length, value, state}
state:  0 empty, 1 occupied, 2 tombstone
```

- `emplace` checks for an existing key first, so the first value wins exactly like `unordered_map::emplace`.
- `erase` marks a tombstone; `rehash` compacts only live slots.
- `find` hashes once, linearly probes, compares length + `memcmp` against the arena, and returns `const int*`.
- Capacity is a power of two with a 0.7 load factor; `reserve(n)` sizes for `n` keys.

Both `vocab_` and `merge_rank_` use the same class. The six call sites (`emplace`, `find`, `erase`, `empty`) moved from iterators to `const int*`; the rest of the tokenizer is unchanged.

## 2. Predicted gain, with the math

From the source-cache cycle:

| component | estimate |
|---|---:|
| lazy tokenizer total | ~103 ms |
| map construction (node alloc + hash + copy) | ~65–80 ms |
| everything else (pieces, control, byte alphabet) | ~25–35 ms |

A flat index with no per-node allocation was predicted to cut map construction to ~25–35 ms:

```text
warm tokenizer ≈ 30 ms (other) + 30 ms (flat maps) ≈ 60–70 ms
speedup ≈ 103 / 65 ≈ 1.4–1.6x
```

The falsifier used the same **1.25x** construction bar as the source-cache cycle, so the two results are directly comparable.

## 3. Falsifier

- **KEEP** only if all hold:
  1. tokenizer construction is at least **1.25x** faster on interleaved quiet-box pairs;
  2. prompt ids and generated ids are identical on the fixed fixtures and thinking set;
  3. the full suite stays green, including direct tests for duplicate, erase/tombstone and rehash.
- **KILL** on any id divergence, test failure, or a median below 1.25x.

Result: ids and tests passed; speed failed at 1.22x combined median.

## 4. Measured numbers

Baseline and spike binaries: same release build, same model (`Llama-3.2-3B-Instruct-Q4_K_M`), same prompt, `--vocabulary lazy --n 1 --threads 6`. Baseline and spike launched back-to-back per pair.

### Set 1 (12 pairs)

| | baseline tokenizer | flat-index tokenizer | ratio |
|---|---:|---:|---:|
| median | 126.2 ms | 101.4 ms | **1.245** |
| best | 108.2 ms | 90.0 ms | **1.202** |

### Set 2 (10 pairs, box more loaded)

| | baseline tokenizer | flat-index tokenizer | ratio |
|---|---:|---:|---:|
| median | 116.3 ms | 96.5 ms | **1.205** |
| best | 101.8 ms | 86.5 ms | **1.177** |

### Combined

- 22 paired ratios; **20/22 faster**, one at 0.966 (flat slower), one at 1.065;
- median ratio **1.221**;
- the absolute construction time falls only ~20–25 ms, not the predicted 35–45 ms.

The flat index is consistently a small win, not a headline. The map build was overestimated: the remaining cost is dominated by copying ~400k keys/strings and building `pieces_`, which the index still pays.

## 5. Correctness evidence

- `string_index_matches_the_map_contract` (temporary test, 146/146):
  - duplicate `"alpha"` keeps the first value (1, not 99);
  - `erase("alpha")` makes `find` return null while `"beta"` survives;
  - 1,000 inserted keys force growth and tombstone reuse; every even key is absent and every odd key returns its value;
  - `"beta"` remains findable after the rehash.
- Thinking set on the spike binary: promptids and generated ids **IDENTICAL** to baseline for:
  - "The quick brown fox jumps over the lazy dog. Explain why." (32 tokens);
  - "Write a haiku about circuits dreaming." (24 tokens);
  - 東京は日本の首都です。×12 (32 tokens).
- Baseline suite after revert: **145/145**.

## 6. Fidelity and gates

- **Numerics**: no model arithmetic touched; no fidelity owed.
- **Kernel bench**: no kernel touched; no kernel A/B owed.
- **Build**: `cmake --build --preset release` green (54/54).
- **Tests**: 146/146 with the direct index test; 145/145 after revert.
- **Project gate**: `python aikit.py check aegean-tge` → **4/4 gates passed** (test 16.2 s, hygiene, guidelines, secrets) after revert.

## 7. Reimplementation recipe (exact, for the record — kill-grade alone)

### 7.1 `include/tge/models/tokenizer.hpp`

Add `<string_view>`, remove `<unordered_map>`, and define `StringIndex` before `Tokenizer`:

```text
class StringIndex {
 public:
  void reserve(size_t count);
  void emplace(string_view key, int value);
  void erase(string_view key);
  bool empty() const;
  size_t size() const;
  const int* find(string_view key) const;
 private:
  struct Slot { uint64 hash; uint32 offset, length; int value; uint8 state; };
  static uint64 hash_of(string_view);
  static size_t next_capacity(size_t);
  bool key_equals(const Slot&, string_view) const;
  size_t find_slot(string_view, uint64 hash) const;
  void rehash(size_t capacity);
  string arena_;
  vector<Slot> slots_;
  size_t size_ = 0, tombstones_ = 0;
};
```

Replace `unordered_map<string,int> vocab_, merge_rank_` with `StringIndex vocab_, merge_rank_`.

### 7.2 `src/models/tokenizer.cpp`

Implement:

```text
hash_of: FNV-1a 64-bit over the key bytes
next_capacity: power of two with capacity*0.7 >= count
reserve(n): rehash(next_capacity(n)) if n exceeds the current 0.7 load
find_slot: probe hash & mask; stop at state 0; match state 1 + hash + length + memcmp
find: return &slots_[index].value or nullptr
emplace: if find(key) return; grow if (size+tombstones+1)*10 >= capacity*7;
         append key to arena; insert at first tombstone or empty; first value wins
erase: find slot, state=2, --size, ++tombstones
rehash: move old slots, assign zeroed new slots, reinsert only state==1 entries
```

Change the build calls to `vocab_.reserve(pieces_.size())` and `merge_rank_.reserve(merges->elements.size())` (the class reserves key count, not bucket count). Change the six call sites to `const int*`:

- `ids.push_back(*found)` instead of `found->second`;
- `if (found != nullptr)` instead of `found != end()`;
- `if (single == nullptr)` instead of `single == end()`;
- `scores_[*found]` in the SPM path.

### 7.3 Test to add

`string_index_matches_the_map_contract` as in §5: duplicate first-wins, erase/tombstone, 1,000-key growth, live/dead key sweep, post-rehash lookup.

### 7.4 Why not ship

The contract is correct and the change is safe, but 1.22x on construction (about 20–25 ms, ~8% of total text-prompt startup) does not justify replacing a standard container with a custom hash table. The measured ceiling also disproves the cache cycle's 65–80 ms build estimate.

### 7.5 What would actually move it

Combine this index with the **serialized final table** from the source-cache recipe: write `arena_`, `slots_`, `pieces_`, control tables and flags to the sidecar; on a hit, `memcpy`/resize them and skip hashing and insertion entirely. That removes the ~100 ms rebuild, not just 20–25 ms of it. Expected warm tokenizer ~15–25 ms; target a 1.4–1.6x total startup win. Reopen only as that combined cycle.

## 8. Verdict and disposition

**KILLED on the 1.25x construction bar.** The flat index is correct, behavior-identical and consistently (20/22 pairs) faster, but the gain is 1.22x median, below the bar and below the predicted 1.4–1.6x. It is not worth the custom container on its own. The recipe is retained because it is the necessary first half of the only remaining big tokenizer win: serialized flat table + sidecar.

## 9. Exact commands run (selection)

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 146/146 spike, 145/145 after revert
<baseline> generate --model Llama-3B --prompt "The capital of France is" \
  --n 1 --threads 6 --vocabulary lazy                  → 22 interleaved pairs
<spike>    generate ...                                 → see §4
tge generate ... --print-ids --text --gpu off           → thinking/CJK ids IDENTICAL
git checkout -- include/tge/models/tokenizer.hpp src/models/tokenizer.cpp tests/test_gguf.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
