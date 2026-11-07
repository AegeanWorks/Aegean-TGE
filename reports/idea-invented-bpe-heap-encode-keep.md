# INVENTED: BPE heap encoder — KEEP (192–400x on long pre-tokens, ids identical)

- **Idea**: Replace `Tokenizer::encode_plain`'s O(n²) "rescan every adjacent pair after every merge" loop with a lazy min-heap of merge candidates over a stable doubly-linked symbol list. New invented direction (not in the dossiers); checked against `docs/measurement-log.md` first — the log covers tokenizer correctness and startup parse cost, but no encode-complexity work. The adjacent `encode_spm` already uses a heap; this is the BPE twin.
- **Cycle date**: 2026-11-07
- **Verdict**: **KEEP-GRADE**. 4,096-char single pre-token 141.1 → 0.7 ms (~202x median), 8,192-char 572.1 → 1.5 ms (~381x median); ordinary prose unchanged; all prompt-id and generated-id fixtures byte-identical; 146/146 tests with the new differential.
- **Tree state at close**: spike reverted from `src/models/tokenizer.cpp`, `apps/cli/main.cpp`, `tests/test_gguf.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; the pre-existing 2026-11-04 stash untouched; nothing pushed.
- **Scope note**: the win is tokenizer latency, not prefill/decode throughput. For a 512-token prefill at ~9.8 s, saving 139 ms is ~1.4% of wall time; the tokenizer path itself is 200–400x faster on the inputs that trigger the old quadratic scan.

## 1. Mechanism

`encode_plain` splits text into pre-tokens, maps bytes to the GPT-2 byte-level alphabet, and starts BPE from one symbol per code point. The old loop then:

1. scans all `n-1` adjacent pairs,
2. builds `left + " " + right` for each pair,
3. looks each key up in `merge_rank_`,
4. merges the lowest rank (leftmost on ties),
5. repeats until no pair merges.

That is `sum(k-1) = n(n-1)/2` pair tests and key builds per pre-token: O(n²) string work. Ordinary prose hides it because the splitter emits short words. A long uninterrupted run — DNA, a long identifier, a CJK paragraph, padding whitespace, a minified blob — is one pre-token and pays the quadratic cost.

The replacement keeps symbols as **stable indices** into the encoded piece:

```text
BpeSymbol { start, length, prev, next, version, alive }
```

Because every merge combines adjacent symbols, a merged symbol is still a contiguous byte range of `encoded`; no per-symbol string concatenation is needed. A merge:

- extends the left symbol's `length`,
- relinks `next`/`prev`,
- marks the right symbol dead,
- bumps the left symbol's `version`.

Candidates live in a `std::priority_queue` ordered by `(rank, left index)`, i.e. lowest rank first and leftmost on ties — exactly the old scan's choice. Stale candidates are rejected on pop by `alive`, `version`, and `next == right` checks; a merge pushes only the two newly created neighbour pairs. Work becomes `O(n log n)` heap operations and `O(n)` merge lookups instead of `O(n²)` pair lookups.

Small pieces stay on a linked-list rescan (no heap allocation); the switch is `symbols.size() > 32`.

## 2. Predicted gain, with the math

For a pre-token of `n` symbols, old pair tests are `n(n-1)/2`:

| n | old pair tests | heap candidate pushes (≤) | predicted |
|---|---:|---:|---:|
| 64 | 2,016 | ~190 | ~10x |
| 512 | 130,816 | ~1,530 | ~85x |
| 4,096 | 8,386,560 | ~12,300 | ~680x raw, less after heap/key costs |
| 8,192 | 33,542,144 | ~24,600 | ~1,360x raw |

The estimate is deliberately optimistic: each heap pop/push is O(log n) and each key lookup still hashes up to the symbol length. The falsifier only required 5x, because the constant factors were unknown before silicon.

## 3. Falsifier

Stated before the spike:

- **KEEP** only if all three hold:
  1. prompt ids and generated ids are identical to baseline on the fixed fixtures and the thinking set;
  2. a 4,096-char single pre-token encodes at least **5x** faster;
  3. ordinary prose (5k chars / 901 tokens) does not regress by more than 5%.
- **KILL** if any one fails.

All three passed; the measured margins are below.

## 4. Baseline reproduction (pre-spike tree)

`tge` was given temporary `--encode-only` instrumentation (time `Tokenizer::from_path` and `encode`, then exit; no model load, no prefill). The instrumented binary was built from the untouched tokenizer and the full suite was green (145/145) before the numbers were taken.

| input | chars | tokens | encode ms (3 runs) | median |
|---|---:|---:|---|---:|
| `a`×4096 | 4,096 | 512 | 141.5, 141.1, 134.3 | **141.1** |
| `a`×8192 | 8,192 | 1,024 | 558.2, 586.7, 572.1 | **572.1** |
| prose, 5k chars | 5,000 | 901 | 0.4, 0.4, 0.4 | **0.4** |
| `-`×512 | 512 | 8 | 2.3 (single) | 2.3 |
| `7`×512 | 512 | 171 | 0.0 (single) | 0.0 |
| mixed short probe | 48 | 24 | 0.1 (single) | 0.1 |

The 4,096 → 8,192 scaling (141 → 572 ms for 2x input, ~4x time) reproduces the quadratic signature. Prose is flat because its pre-tokens are short.

## 5. Spike implementation (reverted; recipe in §8)

Temporary changes, all on `master`, never committed:

- `src/models/tokenizer.cpp`: the heap encoder described in §1, threshold 32.
- `apps/cli/main.cpp`: `--encode-only` harness and `encode:` timing line.
- `tests/test_gguf.cpp`: `tokenizer_heap_merge_matches_the_scan`.

Build: `cmake --build --preset release` → 54/54 targets.
Tests: **146 cases, 0 failed** (145 baseline + the new differential).

## 6. Measured numbers

### 6a. Encode A/B (same binary except the tokenizer loop)

| input | baseline median | spike median | speedup (median) |
|---|---:|---:|---:|
| `a`×4096 | 141.1 ms | **0.7 ms** | **~202x** |
| `a`×8192 | 572.1 ms | **1.5 ms** | **~381x** |
| prose 5k | 0.4 ms | 0.4 ms | 1.0x |
| `-`×512 | 2.3 ms | 0.1 ms | ~20x (single samples) |
| mixed short | 0.1 ms | 0.1 ms | 1.0x |

Spike runs: 4,096 → 0.7, 0.7, 0.9; 8,192 → 1.4, 1.5, 1.5; prose → 0.4, 0.4, 0.6. The prose 0.6 sample is box noise; medians match.

### 6b. Prompt-id parity

Fixtures captured from the pre-spike binary with `--encode-only --print-ids`, then diffed against the spike:

| fixture | result |
|---|---|
| `a`×4096 (512 ids) | IDENTICAL |
| `a`×8192 (1,024 ids) | IDENTICAL |
| prose 5k (901 ids) | IDENTICAL |
| mixed (`x = y + z [i] .com 12345 <|im_start|>user ÇA va? 東京`) | IDENTICAL |
| `-`×512 (8 ids) | IDENTICAL |
| `7`×512 (171 ids) | IDENTICAL |

### 6c. Thinking set (fixed + creative)

Llama-3.2-3B-Instruct-Q4_K_M, `--threads 6 --gpu off`, `--print-ids --text`:

| prompt | n | prompt ids | generated ids |
|---|---:|---|---|
| “The quick brown fox jumps over the lazy dog. Explain why.” | 32 | IDENTICAL | IDENTICAL |
| “Write a haiku about circuits dreaming.” | 24 | IDENTICAL | IDENTICAL |
| 東京は日本の首都です。 ×12 | 32 | IDENTICAL | IDENTICAL |

The generated ids matching is by construction once prompt ids match (same binary, same engine path), but it was checked on a rebuilt baseline, not assumed.

### 6d. Differential test

`tokenizer_heap_merge_matches_the_scan` builds a synthetic 65-piece vocabulary (`a`×1..64 plus `b`) and a merge ladder `a a`, `aa aa`, `aaaa aaaa`, ... `a`×32 + `a`×32. It encodes every run length 1..128 of both `a` and `b` and compares the tokenizer against an in-test, pre-spike O(n²) reference scan. All 128 lengths agree on both alphabets. This covers the heap threshold (33+), the merge ladder, leftmost tie order on identical ranks, and the no-merge heap case.

### 6e. End-to-end note

The spike binary ran `a`×4096 / `--n 2`: encode 0.7 ms, prefill 512 tokens in 9,755 ms (52.5 tok/s), decode 1 token in 164 ms. Baseline encode was 139 ms. Prefill dominates by ~70x on this model, so **no end-to-end throughput claim is made**; the win is the 139–560 ms of tokenization that precedes prefill on long-run prompts.

## 7. Fidelity and gates

- **Numerics**: none touched. No weights, activations, KV, or sampling arithmetic changed; `tge fidelity` is not owed. The tokenizer's output is exact by the id parity above.
- **Kernel bench**: none touched. No kernel A/B owed.
- **Build**: `cmake --build --preset release`, green, both baseline and spike.
- **Tests**: baseline after revert **145/145**; spike **146/146** (the extra differential).
- **Project gate**: `python aikit.py check aegean-tge` → **4/4 gates passed** (test 21.0 s, hygiene, guidelines, secrets) on the reverted tree.

## 8. Reimplementation recipe (exact)

Files to touch:

- `src/models/tokenizer.cpp` — `Tokenizer::encode_plain` (currently starts line 585).
- `tests/test_gguf.cpp` — after `tokenizer_encodes_with_merges` (line 171).
- Optional but useful: `scripts/measure_startup.py` already parses `encode:` if the CLI prints it; or keep the temporary `--encode-only` harness out of the shipped tree and measure through a test binary.

### 8.1 Data structures (anonymous namespace in `tokenizer.cpp`)

```text
BpeSymbol {
  size_t start; size_t length;
  int prev; int next;
  uint32_t version; bool alive;
}
BpeCandidate {
  int rank; int left; int right;
  uint32_t left_version; uint32_t right_version;
}
BpeCandidateIsLater:  // priority_queue comparator
  rank differs  -> left.rank > right.rank
  else          -> left.left > right.left
constexpr size_t kBpeHeapThreshold = 32;
```

`left` is the symbol's stable original index; adjacent symbols always have increasing index, so `(rank, left)` reproduces the scan's leftmost tie break.

### 8.2 Build the symbol list

Replace `std::vector<std::string> symbols` with `std::vector<BpeSymbol>`, one per code point, reserving `encoded.size()`. Set `start`/`length` from the `next_codepoint` walk, set `prev` to the previous symbol, and set the previous symbol's `next` to the new index. Leave the last `next` at -1.

### 8.3 Rank lookup and merge

```text
rank_of(left, right):
  key.assign(encoded, symbols[left].start, symbols[left].length);
  key.push_back(' ');
  key.append(encoded, symbols[right].start, symbols[right].length);
  find key in merge_rank_; return -1 when absent.

merge_at(left, right):
  symbols[left].length += symbols[right].length;
  symbols[left].next = symbols[right].next;
  if (symbols[left].next >= 0) symbols[symbols[left].next].prev = left;
  symbols[right].alive = false;
  ++symbols[left].version;
```

Use one reusable `std::string key` per `encode_plain` call; do not allocate per candidate.

### 8.4 Heap path (`symbols.size() > 32`)

1. Seed the heap with every adjacent pair for which `rank_of >= 0`, storing both symbols' current `version`.
2. Pop the best candidate. Reject if either symbol is dead, either version changed, or `symbols[left].next != right`.
3. Record `before = symbols[left].prev` and `after = symbols[right].next`; call `merge_at(left, right)`.
4. If `before >= 0`, push `(before, left)`; if `after >= 0`, push `(left, after)`.
5. Repeat until the heap is empty. Dead/stale entries need no eager removal.

### 8.5 Scan path (`symbols.size() <= 32`)

Walk the linked list from index 0; for each `index` with `next >= 0`, compute `rank_of(index, next)` and keep the first strictly-smaller rank. Merge that pair and repeat. This is the old algorithm with stable links instead of vector erase; it preserves exact tie order and avoids heap overhead for ordinary words.

### 8.6 Emit ids

Walk the linked list from index 0 via `next`. For each symbol, construct `std::string(encoded, start, length)` and look it up in `vocab_`; on a miss, keep the existing per-code-point fallback and its `invalid_argument` on a truly absent character. Do not change the fallback order.

### 8.7 Test to add

`tokenizer_heap_merge_matches_the_scan`:

- Build a synthetic GGUF with pieces `a`×1..64 plus `b`, and merges `a a`, `aa aa`, `aaaa aaaa`, `a`×8 + `a`×8, `a`×16 + `a`×16, `a`×32 + `a`×32 (ranks in that order).
- Write an in-test `reference_bpe` that runs the old vector-of-strings scan over the same pieces/merges.
- For lengths 1..128, assert `tokenizer->encode(a×n) == reference_bpe(a×n)` and the same for `b×n` (no merges).
- This crosses the 32-symbol threshold, exercises the merge ladder and tie order, and covers the heap with no eligible pairs.

### 8.8 Gate to reproduce the numbers

1. `cmake --build --preset release`
2. `build/release/bin/tge_tests.exe` → 146/146 with the new test.
3. Capture prompt ids on the fixtures and thinking set with `--print-ids`; diff baseline vs candidate.
4. Time `encode` for `a`×4096 and `a`×8192; require ≥5x, no prose regression.
5. `python aikit.py check aegean-tge` from the AIKit folder → 4/4.

## 9. Verdict and disposition

**KEEP-GRADE.** The change is behavior-preserving on every measured path, removes an O(n²) scan with a standard lazy-heap BPE algorithm, and turns a 139–560 ms tokenizer stall on long pre-tokens into sub-2 ms. It is independent of the model, KV format, attention schedule, and kernels; it belongs in the tokenizer, not behind a flag.

The only honest caveat is scope: prose and short prompts were already fast, and prefill dominates long-prompt wall time. The win is real in the tokenizer phase (and in any caller that tokenizes without immediately prefilling), not a decode/prefill throughput claim.

## 10. Exact commands run (selection)

```text
cmake --build --preset release
.\build\release\bin\tge_tests.exe                  # 145/145 baseline; 146/146 spike
tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --prompt <fixture> --encode-only --print-ids     # temporary harness, spike only
tge generate --model ... --prompt <thinking> --n 32 --threads 6 --print-ids --text --gpu off
python aikit.py check aegean-tge                   # 4/4
git checkout -- apps/cli/main.cpp src/models/tokenizer.cpp tests/test_gguf.cpp
git status --porcelain=v1                          # only the ledger/report at close
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
