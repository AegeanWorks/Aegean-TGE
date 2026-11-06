# IDEA-1553: Paged KV with Bounded Prefetch and mmap Advice — SPLIT (advice-for-RSS killed as self-contradictory; paged half redirects to 1517/1585; Q4xQ8 arm already ships)

- **Idea ID**: IDEA-1553 (dossier `kilo-sweep-2026-11-03-b4.md`: MemVanta `mmap_file.cpp` (MAP_PRIVATE, MADV_RANDOM/SEQUENTIAL/WILLNEED/DONTNEED, page-aligned advise_range, POSIX-only) + `gguf_kernels.cpp` Q4_0/Q8_0 AVX2 dots; port paged KV + bounded prefetch accounting + mmap advice for lower peak RSS at same-model parity on CPU)
- **Cycle date**: 2026-11-06
- **Verdict**: SPLIT three ways. (a) mmap-advice/bounded-prefetch FOR LOWER PEAK RSS: KILLED — the falsifier demands ≥30% peak-RSS cut at parity, and parity forbids it: same decode speed requires the same weight residency, and the same residency is the same peak RSS (arithmetic, not a tuning gap). (b) Paged KV: REDIRECTED, not re-judged — design surface ships as `BlockTable` + kv-block-layout headers (measurement-log §§65–66) and engine wiring is tracked under IDEA-1517 (staged recipe, stage-1 headroom proceed) + IDEA-1585 (block-table track); building it here would fork those tracks. (c) Q4xQ8 AVX2 dot within epsilon of scalar: SUPERSEDED — ships as the kq8 integer prefill path (KL floor 1.4e-11) with Q6_K AVX2 behind it, so the dossier's third falsifier arm ("Q6_K scalar-only dominates") is already false. No spike built (nothing a spike could change: (a) is arithmetic, (b) belongs to another track, (c) ships). No branch, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched), `master` ahead of `origin/master` by 52 before this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered, per half)

Port MemVanta's file layer (page-aligned `advise_range` over mmap with RANDOM/SEQUENTIAL/WILLNEED/DONTNEED) plus bounded prefetch accounting plus a paged KV cache, with Q4_0/Q8_0 AVX2 dots under runtime dispatch, for lower peak RSS at same-model parity. Must-be-true: Q4xQ8 dot matches scalar within epsilon on TGE shapes; RSS win reproduces on the TGE host; Windows port of the mmap file via CreateFileMapping (source is POSIX-only — "conditional pass" in the dossier's own words).

## 2. Predicted gain — and the per-half disposition

| half | disposition | grounds |
|---|---|---|
| advice + bounded prefetch → −30% peak RSS at parity | KILLED | §6: parity ⇒ same residency ⇒ same peak; DONTNEED on weights re-faults per token and destroys the §26c fault-up-front win; bounded prefetch shapes load timing only (§26c: same total bytes) |
| paged KV | REDIRECT to 1517/1585 | `BlockTable` + `kv_block_layout` headers ship (§§65–66); wiring lives in the 1517 staged recipe + 1585 track |
| Q4xQ8 dot within epsilon | SUPERSEDED | kq8 integer path ships as the prefill default (KL 1.467e-11 floor); Q6_K AVX2 ships (§28j); third falsifier arm already false |

## 3. Falsifier (from the dossier) — resolved per arm without building

"Per-token KL vs scalar oracle above epsilon" — moot: the shipped kq8 path prices at 1.467e-11 mean KL, ~eight orders inside any epsilon the source's scalar fallback would set; an SHq-style re-proof is not owed for a shipped path. "RSS reduction below 30 percent" — fails by construction for the advice half (§6): there is no advisory flag that keeps weights resident for 97%-of-ceiling decode (§28ak/§62: matvec 94–97% of every step) while cutting the resident peak 30%; the only mechanisms that cut peak (evicting weights, smaller weights) break parity by definition. "Q6_K scalar-only path dominates mix" — already false on today's tree (Q6_K AVX2 + kq8 integer both ship; the mix is vectorized everywhere it executes, §28l). A spike could only re-measure these settled facts.

## 4. File anchors (current `master`, read but untouched)

- `src/models/gguf.cpp:434` (CreateFileMappingW — the Windows mapping the dossier's POSIX source lacks; the "conditional pass" port it requires), `:597` (PrefetchVirtualMemory — the shipped whole-file fault-up-front, §26c).
- `include/tge/memory/io_advice.hpp` (PrefetchBudget + range math, no OS calls — §68 helpers, wiring gated on the same ≥30% RSS cut this report kills).
- `include/tge/models/block_table.hpp`, `include/tge/models/kv_block_layout.hpp` + `tests/test_block_table.cpp` (§§65–66: allocate/share/release refcount, FNV-1a prefix hash, K-column-major/V-row-major gather; Qwen2 cache untouched by design).
- `src/models/qwen2.cpp` batch routing (kq8 integer for Q4_K/Q6_K prefill batches, tokens > 1) + `src/cpu/avx2/dequant_batch_kq8_avx2.cpp` (the shipped Q4xQ8 dot).
- `reports/idea-1517-stage1-headroom-proceed.md` (paged staged recipe), measurement-log §§62/65–66/68 (decode attribution, block-table waves, IO helpers).
- Greps: `MADV|DONTNEED|madvise` → zero matches (POSIX mechanism has no Windows carrier); `advise_range` wiring → zero (helpers only); `BlockTable` engine wiring → zero (headers + tests only, by design).

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Read the dossier body against the current tree; split the bundle per half (§2).
2. Carrier greps (§4): POSIX advice zero, wiring zero, block-table headers present-but-unwired (as designed), kq8/Q6_K-AVX2 shipped.
3. Gates for the record on the unmodified tree: `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141; `python aikit.py check aegean-tge` → test + secrets pass, hygiene + guidelines fail pre-existing — with one honest flake: the aikit `test` gate failed once at 2.4 s (too fast for a real run; the known link-lock/AV-scan race of §§56/63 with no code behind it) and the immediate direct `scripts\test.cmd` rerun rebuilt everything and passed. ctest + 141/141 green in both attempts.
4. No fidelity run owed (shipped paths already priced; no new arithmetic); no bench A/B owed (no second path); no thinking-set diff owed (same binary).

## 6. Measured numbers — none owed (settled facts + one flake, both logged)

No benchmark belongs to this split: the kill arm is arithmetic (§6 below), the redirect arm is tracked elsewhere, the supersede arm already ships with numbers on record (kq8: 1.58–2.22x kernel, 1.6–1.83x end-to-end prefill, KL floor — §§58/61). Gate record: build 54/54, ctest pass, 141/141, aikit 2/4 modulo the one flaked-then-green test gate (§5.3).

Peak-RSS arithmetic (the kill, no run needed): decode streams ~all weights per token (0.5B-Q8_0 500.8 MiB/token per `tge plan`; Llama-3B 1.57 GiB/token) at 94–97% matvec share. Resident peak ≈ weights + KV + scratch by necessity of that streaming. Advice flags do not evict what the next token immediately re-reads; DONTNEED-ing weights converts every decode step into page faults (the exact cost §26c paid 489 ms up front to remove). Bounded prefetch changes WHEN pages fault (load shape), never WHETHER (same total bytes — §26c). Hence no advisory construction reaches −30% peak at parity; the falsifier's two constraints contradict each other.

## 7. Full reimplementation recipe (deliberately NOT recommended — and what each half would need)

Advice half: do not build — revival requires refuting §6 arithmetic (a residency argument showing 30% of peak RSS is never re-read at parity), not a flag wrapper. Paged half: do not build here — follow the 1517 staged recipe and the 1585 block-table track (next: engine wiring of `BlockTable` into the Qwen2 KV cache with the 8x32 ≥58.3 t/s + identical-ids gates). Q4xQ8 half: nothing to build (ships). To re-verify this split: re-run the §4 greps (POSIX advice still zero, wiring still zero-or-tracked, kq8 still default) and confirm no RSS-cut claim has appeared without an eviction mechanism behind it.

## 8. Verdict and why

**SPLIT: advice-for-RSS killed (self-contradictory falsifier + zero POSIX carrier on a Windows-native tree), paged KV redirected to its existing tracks (1517/1585), Q4xQ8 dot superseded (ships with KL floor).** The bundle's only genuinely open work (paged-KV engine wiring) already has owners and gates; the loop's honest output is the map, so no future cycle re-treads it from this dossier.

## 9. Exact commands run and their results

- Carrier greps (§4): MADV/DONTNEED zero; advise wiring zero; block-table headers present, engine wiring zero (by design); kq8 + Q6_K-AVX2 shipped.
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141.
- `python aikit.py check aegean-tge` → test gate flaked once (2.4 s race, §§56/63 precedent), direct `scripts\test.cmd` rerun → pass; secrets pass; hygiene + guidelines fail pre-existing.
- No branch, no edits, no bench, no fidelity, no push.

(End of file.)
