# IDEA-1501: L2-Blocked Online-Softmax GQA with Quantize-on-Write KV — SUPERSEDED (chunked GQA + qow KV ship)

- **Idea ID**: IDEA-1501 (dossier `kilo-sweep-2026-11-03-b2.md`: onnxruntime tiled GQA + split-K decode + quantized-KV path with fused dequant, MoveKvCacheRows windowed ring; claim: long-context wins at idle-thread shapes)
- **Cycle date**: 2026-11-06
- **Verdict**: SUPERSEDED — the operative mechanisms ship: span-chunked online-softmax GQA with per-chunk merge states (split-K over kv_heads × span chunks), per-block quantize-on-write i8 KV with fused dequant on read, bit-parity gated by differentials + fidelity. The windowed-ring remainder is not needed (fixed slots fit RAM several times over). No spike built (duplicating shipped machinery); no branch, no code touched.
- **Tree state at close**: `git status` clean (no diff), stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 37 after this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Tile GQA attention into blocks with online softmax (per-chunk maxima/sums,
merged reduction), split-K parallelism over heads and span chunks, KV
quantized at write time and fused-dequantized on read, windowed ring cache
management.

## 2. Predicted gain — and where the tree already collects it

- Tiled online-softmax GQA: `Qwen2::attention_gqa` splits each decode head
  into up to `kMaxGqaChunks` span chunks (`threads*2/kv_heads`-sized),
  computes per-chunk softmax states (`group_maxima/sums/acc`), merges via
  `merge_gqa_chunks` — online-softmax with split-K parallelism, exactly the
  dossier shape.
- Quantize-on-write KV + fused dequant: `store_kv_q8` quantizes per-32-block
  at every store site; read paths dequantize on the fly (`q8_dot` with
  per-block scales). No fp32 KV materialization on the i8 path.
- Bit-parity: per-family differential tests vs scalar oracle + fidelity
  harness (i8 KL 3.2e-4, top-1 held).
- Windowed ring: absent — and correctly so. Fixed per-slot caches already
  cover the largest measured contexts in RAM several times over (16 GiB box,
  biggest model 2.23 GB); a ring/eviction layer would add machinery for zero
  measured pressure (same finding as the 1382 MoE kill).

## 3. Falsifier (from the dossier, adapted) — does not trip

Bit-parity vs scalar (differentials green), fidelity oracle pricing (priced),
long-context wins at idle-thread shapes (GQA chunking IS the idle-thread
parallelism: `threads*2/kv_heads` chunks keep workers fed on short spans).

## 4. File anchors (current `master`, read but untouched)

- `src/models/qwen2.cpp:1294-1343` (`attention_gqa` chunking + q8/f32 read
  routing), `merge_gqa_chunks` (online-softmax merge).
- `src/cpu/scalar/dequant.cpp` (`store_kv_q8`), `src/cpu/avx2/*q8*` (fused
  read), `tests/test_dequant.cpp` + `test_attention.cpp` (differentials).

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Mapped each operative sub-mechanism to shipped code (§2).
2. Gates for the record: build clean, `ctest` pass, `aikit.py check` 2/4
   (test + secrets pass; hygiene + guidelines fail pre-existing).
3. No thinking-set diff owed: same binary, zero behavior change.

## 6. Measured numbers (cited — the shipping receipts)

GQA chunking ships with the thread-scaled chunk formula; i8 KV prefill +36%
vs float (1458 cycle, 6.27 vs 4.61 GB/s); i8 fidelity KL 3.2e-4, top-1 held.
No new measurement is owed by a superseded idea.

## 7. Open residue (none from this idea)

The windowed ring belongs to a pressure regime this box never reaches (fits
RAM 3x over). If contexts ever exceed RAM, ring/eviction becomes its own
dossier idea — until then it is correctly absent.

## 8. Verdict and why

**SUPERSEDED.** Tiled online-softmax GQA with quantize-on-write KV is this
engine's attention architecture, not a task. The only unshipped fragment
(windowed ring) has no pressure to answer to.

## 9. Exact commands run and their results

- `cmake --build --preset release` → 54/54; `ctest` → pass.
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits, no push.
