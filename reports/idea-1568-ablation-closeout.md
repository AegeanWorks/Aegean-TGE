# IDEA-1568: Portability Ablation Protocol — CLOSED (the run the deep-check asked for; triage CONFIRMED with two refinements)

- **Idea**: IDEA-1568 (dossier `scout-2026-11-goal-finalpapers.md`, doc `docs/ablation-1568.md`, log §64): the 4-family x86 batch-one ablation the deep-check explicitly queued ("no 4-family x86 batch-one ablation run; run first to triage pruning vs speculative"). This cycle IS that run: KV-pruning dense-vs-sparse, speculative on/off at batch-one, PTQ f16 re-price, ported-kernel record review — same box, same window, back-to-back per the protocol's own rules.
- **Cycle date**: 2026-11-06
- **Verdict**: CLOSED — triage CONFIRMED with two refinements. (1) KV pruning: stays OPT-IN and model/shape-gated, and the gate narrows: at 2k-token prefill sparse ATTN is slower on both models (0.5B +35%: 1024→1403 ms; Llama-3B +13%: 7731→8750 ms — gather+index overtake the halved dots), while sparse DECODE wins on the big model (+21%: 7.57→9.13 t/s, Llama-3B). The protocol's kill criterion does not fire (nothing to remove — already opt-in, default dense), but "kept" now means decode/long-context-large-model, not prefill. (2) Speculative: does NOT vanish at batch-one (the Apple outcome): +43% on repetitive text (75.84→108.38 t/s, 5.50 accepted/step), parity (−1.4%, inside noise) on non-repetitive text — kept for decode, correctly never aimed at the prefill gap (prefill identical 272 vs 275). (3) PTQ f16 re-priced in-window: KL 2.021e-06, 1023/1023 — inside the 1.7e-6 precedent class, top-1 held; i8 tier stands on the 2048-closure ladder (f16 best at 2048). (4) Ported kernels: integer lane stays 0-for-5 (§12×2, §28i, §34, §44) with the kq8 exception intact and re-verified this session (9.248e-12 on Q8_0, 1.006e-11 on Q4_K_M full-corpus); ported-GEMM/MoE/trunk/Rust/SHq halves all closed with carriers absent or superseded — the "distrust, verify natively" stance holds without modification. No code changed (harness-only, per the protocol's "no engine change in this step"); nothing to revert.
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). No branch created (no code to spike — measurement round, precedented §§43/45). `master` ahead of `origin/master` by 74 before this report. Nothing pushed (human gate).

## 1. Mechanism (the run, not a code change)

Four arms, one window, back-to-back pairs: (a) 2000/2050-token prefill + short decode, dense vs `--sparse`, on 0.5B-Q8_0 AND Llama-3B-Q4_K_M; (b) story + repetitive 70-token prompt, `--speculative` vs not, 0.5B-Q8_0; (c) `fidelity --kv f16` (0.5B-Q8_0, arch corpus, ctx 1024); (d) ported-kernel record audit against the report library (no new runs — analytic, each with its carrier proof on record). Falsifier (the protocol's own kill criteria): speculative 1.66–2.0x vanishes at batch-one (Apple outcome) → deprioritize; pruning ≤20% back-to-back or KL/top-1 regress → reject to dense+GQA; ported kernels need async-copy/serving or PTQ varies beyond harness noise → non-transferable; any family breaking greedy-id identity or the fidelity gate → rejected for that family.

## 2. Predicted gain — triage, not speedup

The run was owed, not the gain: close the deep-check's open item with same-window numbers and either confirm the standing triage or trigger a kill criterion. Outcome: confirmed, with the pruning-gate narrowed and everything else unchanged.

## 3. Falsifier outcome — no kill criterion fires; two refinements recorded

- Speculative: no vanish, no regression beyond noise → criterion NOT triggered; stance holds (decode-side, repetition-gated; the 1.66–2.0x band from §§71/75 remains prompt-dependent, 5.50 accepted/step measured).
- KV pruning: the ≤20% clause reads as "no win, no mandate" — prefill shows a LOSS at 2k (−10% 0.5B, −7% Llama), decode a split (+21% Llama, −8% 0.5B); fidelity arm clean (KL 2.597e-02/975 recorded §50, ids-gated). Verdict: keep opt-in + gate-narrowed (decode/long-context/large-model), default dense everywhere. No rejection (already the status quo).
- Ported kernels: nothing new to port; every named half has a verdict with carriers (see §4).
- PTQ: f16 holds the gate (2.021e-06, 1023/1023); i8 holds its conditional (size-ladder); SHq stays killed.

## 4. File anchors (read, not written — except this report)

- `docs/ablation-1568.md` (the protocol executed here), `docs/measurement-log.md` §64 (protocol landing).
- `src/models/qwen2.cpp` gather sites + `uses_sparse` (the attn-phase cost measured in §6.1 lives here).
- `apps/cli/main.cpp` (`--sparse`, `--speculative`, `--kv`).
- Reports library standing behind arm (d): 1325 (F32-GEMM bench-only), 1539-40 (ships), 1558/1458 (killed), 1451 (scored killed with measurement), 1452 (superseded), 1519 (MXFP4/MoE zero), 1382/1386 (no exporter), 1553/1585/1586 (split), 1569 (autotune bench-only + sub-1% TwoMatMul), 1582 (Rust dead, int-dot ships), 1559 (SHq dead), 1394 (quantize-once integrated), error-split round (kq8 full-corpus floors re-verified this session: 9.248e-12 on Q8_0, 1.006e-11 on Q4_K_M), 2048-closure (f16 best at 2048, i8 corrected).

## 5. What was done (measurement round — no code, no branch)

1. Gates first: `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141; `python aikit.py check aegean-tge` → test gate flaked once at ~2.9 s (the §56/63 race signature — ninth occurrence), immediate direct `scripts\test.cmd` rerun green; secrets pass; hygiene + guidelines fail pre-existing. Wall 59.20 healthy bracketing.
2. Arm 1 (§6.1): fox ×45 first (spans 450 < 516 dense-fallback — caught as vacuous, re-ran fox ×200/×205 for engaged comparisons).
3. Arm 2 (§6.2): story + repetitive prompt, speculative on/off.
4. Arm 3 (§6.3): f16 fidelity re-price.
5. Arm 4 (§6.4): record review — greps re-confirmed zero-int-dot-carrier claims where kill verdicts rest on them (no new runs needed; the reports hold the measurements).
6. Thinking set: story ids byte-identical to the baseline fingerprint; creative prompt fluent.
7. No branch (nothing to spike), no edits, nothing to revert; tree verified clean + stash unchanged at close.

## 6. Measured numbers (single window, back-to-back pairs; box drift noted per pair)

6.1 KV pruning (prefill attn ms is the gated metric; matvec drift tracked as the load control):
- 0.5B-Q8_0 @2000: dense prefill 312.0 (matvec 3812, attn 1024, norm 1527) vs sparse 281.1 (4104, 1403, 1557); repeat interleaved S/D: sparse2 291.2 (3833, 1428) vs dense2 284.8 (4344, 1069). Attn-phase sparse/dense ≈ 1.35x consistently (load-independent); matvec drifted box-wide (+8–14% across the window) and affects both arms.
- Llama-3B-Q4_K_M @2050: dense prefill 51.8 (29330, 7731, 2263), decode 7.57 vs sparse 48.2 (31181, 8750, 2372), decode 9.13 (+21%).
- Reading: static-stride sparse loses the prefill-attn phase at 2k on both models (gather+index > halved dots); its batch-one value is decode-side on large models. Matches the shipped posture exactly (default dense; `--sparse` opt-in with its own falsifier).

6.2 Speculative (0.5B-Q8_0, batch-one): story (4-tok prompt, 31 decode) 79.34 vs 78.23 t/s (−1.4%, noise); repetitive 70-token prompt 75.84 vs 108.38 t/s (+43%, 5.50 accepted/step, 13 emitted). No Apple-style regression anywhere; the repetition gate from §51 stands.

6.3 PTQ: f16 fidelity 2.021e-06 mean, 1.609e-04 worst, 1023/1023 top-1 — inside the 1.7e-6 precedent, agreement held.

6.4 Ported kernels: 0-for-5 integer record re-confirmed by reference (§12×2, §28i, §34 at KL 8e-3 + 2 flips, §44 at −1.7x loss) with the kq8 exception intact (default-on; this-session full-corpus floors: 9.248e-12 on Q8_0, 1.006e-11 on Q4_K_M). Zero-carrier verdicts (MXFP4, MoE, SHq, Rust, pread/O_DIRECT, GGUF writers, full-KV F32 GEMM callers) re-grepped clean where load-bearing. No new runs owed — the reports ARE the measurements.

## 7. Full reimplementation recipe (runbook, not code — re-run to re-verify)

1. Gates: build 54/54, ctest pass, 141/141; wall 48+ with story 8–12 t/s (the §45 health gate).
2. Arm 1: fox ×200 (0.5B) / ×205 (Llama-3B), `--context 4096`, `--n 8`, dense then `--sparse`, back-to-back; expect sparse attn-phase ≥ dense (the narrowed gate — flip only if sparse wins a shape).
3. Arm 2: story + fox×7, `--speculative` vs not; expect tie on story, win on repetitive (acceptance printed: ×/step).
4. Arm 3: `fidelity --kv f16` (0.5B-Q8_0, arch corpus, ctx 1024); expect ~2e-6, 1023/1023.
5. Arm 4: re-grep the zero-carrier list; any newly-appeared carrier reopens its idea, everything else stands.
6. Kill-trigger watch (the protocol's own): speculative regression beyond noise → deprioritize; pruning KL/top-1 regress → reject to dense; ported kernel needing async-copy/serving → non-transferable. None triggered this round.

## 8. Verdict and why

**CLOSED: the run the deep-check asked for, triage CONFIRMED with the pruning gate narrowed.** Every kill criterion was given a live chance to fire and none did — which is itself the result: the standing map (KV-pruning opt-in + model/shape-gated; speculative decode-side + repetition-gated; integer lane distrusted except priced kq8; ported kernels judged per-carrier; PTQ gated by fidelity) survives contact with a fresh same-window battery, with one honest correction (sparse prefill at 2k loses on both current models — the "kept" in the triage always meant decode/long-context, and now the numbers say so explicitly). No future cycle needs to re-litigate the map; future cycles only need to watch the three kill triggers in §7.6.

## 9. Exact commands run and their results

- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141; wall 59.20.
- `python aikit.py check aegean-tge` → test gate flaked once (~2.9 s), direct `scripts\test.cmd` rerun → pass; secrets pass; hygiene + guidelines fail pre-existing.
- Arm 1: fox×45 pair (caught vacuous — under the 516 sparse threshold, both dense); fox×200 pair + interleaved repeat (0.5B: dense 312.0/284.8 vs sparse 281.1/291.2; attn 1024/1069 vs 1403/1428); Llama ×205 pair (dense 51.8/7.57 vs sparse 48.2/9.13).
- Arm 2: story 79.34 vs 78.23; repetitive 75.84 vs 108.38 (5.50/step).
- Arm 3: f16 fidelity → 2.021e-06, 1023/1023.
- Arm 4: grep re-verification of zero-carrier claims; story ids byte-identical; creative prompt fluent.
- No branch, no edits, no bench beyond the battery, no push.

(End of file.)
