# IDEA-1582: Pure-Rust GGUF mmap with Int-Dot Tiers and Parity Gate — SPLIT (int-dot + parity ships; Rust rewrite dead: zero .rs files, wrong language)

- **Idea ID**: IDEA-1582 (dossier `kilo-sweep-2026-11-03-b5.md`: frink `frink-gguf` sharded open without loading weights + `ferrox-quant/repack.rs` block_q4_Kx8/block_q8_0x4/cpu_int_dot + parity-verify methodology — "reuse mmap-resident quantized weights with fused dequant inside matmul plus parity verify methodology as tge oracle even though Rust not C++20")
- **Cycle date**: 2026-11-06
- **Verdict**: SPLIT. (a) Int-dot tiers with Q8 activation quant + type routing + parity-verify methodology: SUPERSEDED — ships natively as the kq8 integer prefill path (default on everywhere: generate, fidelity, bench), the kq8d single-row decode variant, VNNI int8, and the differential + fidelity parity discipline (kq8 KL 1.467e-11 floor, VNNI 1.6e-3 priced). (b) Pure-Rust GGUF mmap rewrite: KILLED — zero `.rs` files and zero `Cargo.*` manifests in the repo, no Rust toolchain in the build, wrong language for a C++20 project by the dossier's own concession ("even though Rust not C++20"); every load-bearing piece it names already ships natively (CreateFileMappingW + PrefetchVirtualMemory mmap, block-repack discipline in dequant_batch, int-dot tiers). No spike (dead-idea rule: the live half ships with numbers, the dead half has no carrier to port). No branch, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched), `master` ahead of `origin/master` by 60 before this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered, per half)

Reuse mmap-resident quantized weights (sharded open without loading) with fused dequant inside the matmul (block_q4_Kx8/block_q8_0x4 repacks, cpu_int_dot) plus a parity-verify methodology as the engine oracle — ported from Rust into the C++20 tree.

## 2. Predicted gain — and the per-half disposition

| half | disposition | grounds |
|---|---|---|
| int-dot tiers + Q8 activation quant + type routing + parity verify | SUPERSEDED (ships) | kq8 default-on (generate/fidelity/bench) + kq8d decode variant + VNNI, KL-priced, differential-gated |
| mmap-resident quantized weights (sharded open) | SUPERSEDED (ships) | CreateFileMappingW + PrefetchVirtualMemory, weights never copied (§68/§26c family) |
| Pure-Rust port vehicle | KILLED | zero `.rs` / `Cargo.*` in repo, no Rust toolchain, C++20 project; all named pieces already native |

## 3. Falsifier (from the dossier) — resolved without building

"Q8_0 drifts vs llama.cpp on Llama-3.2-3B Q4_K_M" — moot: parity is held by the native discipline, not the port vehicle (scalar-oracle differentials memcmp-gated everywhere hot; fidelity floors: kq8 1.467e-11, norms/RoPE/SiLU full-program 2.6e-12; story/short ids byte-identical across tiers on every measured round). A Rust reimplementation would re-prove the same parity at a language boundary that adds nothing. "Int-dot still 6x behind on AVX2 host after scalar check" — already false with the sign flipped: int-dot is FASTER than float here (kq8 1.58–2.22x kernel / 1.6–1.83x end-to-end prefill, §§58/61; VNNI 1.9x prefill, §23), not 6x behind — the dossier's fear describes the source's aarch64/AVX2 gap (0 AVX2 sites, issue 152), which the native port closed by construction.

## 4. File anchors (current `master`, read but untouched)

- `apps/cli/main.cpp:111,121` (`--kernel kq8` "default on" help), `:1972/:2475` (`bool kq8 = true` defaults), `:560`,`:2323` (`set_kq8`), `:1296-1297` (`gemv_dequant_kq8_row/batch` dispatch), `:2820-2833` (bench kernel select incl. kq8d).
- `src/models/gguf.cpp:434` (CreateFileMappingW), `:597` (PrefetchVirtualMemory fault-up-front) — the native sharded-open equivalent (map, never copy).
- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp` (block-repack + integer-dot discipline — the native `block_q4_Kx8` analog with KL floor).
- Globs: `**/*.rs` → none; `**/Cargo.*` → none (verified this cycle).
- Precedents: 1519 (MXFP4/MoE zero-carrier kill — same "source ISA/format absent" discipline), 1553/1586 (shim/port splits — ship-what-ships pattern).

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Read the dossier body against the current tree; split per half (§2).
2. Carrier inventory (§4): Rust zero (globs), mmap native, int-dot default-on with prices.
3. Gates for the record on the unmodified tree: `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141; `python aikit.py check aegean-tge` → secrets pass, hygiene + guidelines fail pre-existing, test gate flaked once at 2.5–2.8 s (the §56/63 race signature — fifth occurrence, no code behind it) with the immediate direct `scripts\test.cmd` rerun rebuilding everything and passing. ctest + 141/141 green in both attempts.
4. No fidelity run owed (tiers already priced); no bench A/B owed (no second implementation); no thinking-set diff owed (same binary).

## 6. Measured numbers — none owed (both halves settled on record)

No benchmark belongs to this split: the live half's numbers ship (kq8 speedups + KL floor §§58/61; VNNI 1.9x + 1.6e-3 §23; prefetch timing §26c) and the dead half has no carrier to measure. For orientation: the source's "17 aarch64 sites, 0 AVX2" is precisely the gap the native AVX2 int-dot paths close — porting the Rust would re-cross a bridge already built in C++.

## 7. Full reimplementation recipe (deliberately NOT recommended — both directions closed)

Do not port Rust: there is no Rust to port to (no toolchain, no files, C++20 project). Do not re-prove int-dot parity: the differentials + fidelity floors already gate every hot path. Revive only the union neither half covers — none exists: mmap ✓, repack ✓, int-dot ✓, parity ✓. To re-verify this split: re-run the §4 globs (still zero) and confirm `--kernel kq8` still default-on with its KL floor intact.

## 8. Verdict and why

**SPLIT: int-dot + parity SUPERSEDED (ships with numbers), Rust vehicle KILLED (zero carriers, wrong language, everything it names already native).** The dossier's own concession ("even though Rust not C++20") is the verdict: the methodology transferred, the vehicle didn't need to.

## 9. Exact commands run and their results

- Carrier inventory (§4: `**/*.rs` + `**/Cargo.*` zero; kq8 default-on sites; mmap natives).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141.
- `python aikit.py check aegean-tge` → test gate flaked once (2.5 s race), direct `scripts\test.cmd` rerun → pass; secrets pass; hygiene + guidelines fail pre-existing. No branch, no edits, no bench, no fidelity, no push.

(End of file.)
