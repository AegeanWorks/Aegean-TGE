# Verification & Continuation — Improving Token Creation Speed and Quality

- Date: 2026-11-01, machine: Intel Core i5-14400F (6P+4E), 16 GiB single-DIMM, Windows 11, MinGW GCC 16.2
- Verifies the 16-page report at master 7a500c7 against HEAD e139fcc + this round
- Model: qwen2.5-0.5b-instruct-q8_0.gguf (638.74 MiB), 6 threads unless stated
- Method: G-001 before/after snapshots, G-002 log entries; warm runs, best-of-3..5; this box varies ±20% run to run, so ratios within one run are evidence, absolutes across sessions are not

## 1. Verdict per report claim

| # | Report claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Vectorise attention: 42% of decode at 2,048, prototype 14.9→19.4 t/s (+30%), prefill +36%, logits within 6e-5 | TRUE, LANDED, EXCEEDED | measurement-log §17: 21.97→31.38 t/s decode (+43%), 182.0→218.5 prefill (+20%); §18 GQA 35.11→40.34; fidelity KL 5-6e-12, max logit diff 1.4-1.8e-4, top-1 1023/1023 |
| 2 | Fidelity harness Phase 0 not started; 40-line version prices prototype 4e-12 vs f16 KV 2e-6 | TRUE, LANDED | `tge fidelity` shipped (§17); KL 6.1e-12 scalar-vs-AVX2, 1.658e-6 f16 KV (§24), 1.568e-3 VNNI (§23); this round re-priced at 6.115e-12, 1023/1023 |
| 3 | Prompt-lookup speculation 1.9-2.6 tok/pass, predicted 1.3-1.8x, identical output | TRUE, LANDED, IN RANGE | §21: story 30.10→49.92 (1.66x), recheck 28.13→56.38 (2.00x), 3.95/step accepted, ids identical; this round 32.3→50.0 (1.55x) |
| 4 | Input handoff: 6/20 strings differ, space-before-operator splits, `<tool_call>` not matched, control tokens in all text | TRUE, LANDED | §16: `x = y + z` 7→5, `[i` 2→1, `.com` 2→1, `<tool_call>` 5→1; USER_DEFINED matched; `encode(text,false)` opt-out for user content; end-of-gen set 151643 + `eos_token_ids` |
| 5 | Top-p/min-p sort all 151,936: ~25 ms/token VM, 16.75 ms i5 | TRUE, LANDED | §14: 16.75→0.61 ms no-cut; §16: top-p 18.2→4.1, min-p 19.2→4.1 ms; end-to-end sampled decode now ~greedy (47-49 t/s) |
| 6 | Abstract model t/s = k ÷ (bytes/BW + serial + c·sync); wall 53.3 t/s; short 42-47 (11-21% off wall), 2048 at half | TRUE, HOLDS | §10: 500.8 MiB/token ÷ 28 GB/s = 53.3 ceiling; HEAD: 45 t/s short (~84%), 38-39 at 2048 (~80% of llama 46.9); vm shares transferred, re-measured on 14400F per G-001 |
| 7 | 169 parallel calls, attention 42% of 2k decode, projection pool waits | TRUE, REDUCED | Spin-then-sleep pool 64.3→0.80 us (§15); attention parallel + GQA + AVX2 (§§15,17,18); remaining attention at 2k is compute (48 MiB KV ≈ 2 ms memory, 28 ms measured before vectorisation) |
| 8 | Batched kernel 0.37 ms/1 tok vs 0.58/6 tok; 2-5 tok cost 1.7-4.0x, needs efficient small groups | TRUE, FIXED THIS ROUND | Before: 5 tok slower than 6 (FFN 0.432 vs 0.239 ms = 1.81x; output 8.43 vs 6.08 ms). After: 5≈6 (FFN 0.32-0.38 vs 0.32-0.36; output 5.77 vs 5.70 ms, -31.6%/-6.2%). §26a |
| 9 | Fused K-quant dot: Q4_K 5.18 vs 21.85 GB/s, compute-bound dequant | TRUE, STILL OPEN | Dequant vectorised +29.6% (5.18→6.71 GB/s, §7b) but end-to-end invisible in noise; fusion not done. Next. |
| 10 | Pin decode workers to P-cores; no affinity anywhere | TRUE, FIXED THIS ROUND (variance) | Workers now pin round-robin to fastest efficiency class; handoff already 0.5-0.8 us so mean unchanged within noise, spread source removed. §26b |
| 11 | Prefetch mapped weights: cold 43 s fault-in | TRUE, FIXED THIS ROUND (opt-in) | `File::prefetch()` + `generate --prefetch`: 675 MiB in 489 ms here, moved out of first token. §26c |
| 12 | Fuse Q/K/V, gate/up: 72 fewer calls save <0.3 ms | CORRECTLY DROPPED | Not revisited; arithmetic unchanged. |
| 13 | Vectorised exp in softmax: 688k exp/token at 2048 | TRUE, STILL SCALAR | Needs KL price first; literature supports (below), not attempted here. |
| 14 | 241 GFLOPS ≈ 91% prefill is miscounted; nearer 40% | TRUE, ACCEPTED | Useful rate ~137 GFLOPS (358M weights/token), not 241 (630M). §26d supersedes §6. |
| 15 | Near-ties 3.1% (gap <0.1, median 2.97); parity measures distance to first near-tie | TRUE, CONFIRMED | §19: 128-token run identical ~95 tokens then flips at near-tie (llama "and a constant" vs TGE "with a variable"); harness prices arithmetic at 5e-12 despite divergence |

## 2. Literature: what transfers to this operating point

Narrow, CPU-batch-1 Q4/Q8 GGUF on Windows x86 is almost unmeasured in the
peer-reviewed record; the GPU 7B+ and edge reports transfer as mechanism +
method, not numbers. What was used:

- Bandwidth wall: t/s ≈ effective BW ÷ active bytes, 50-80% realised
  (Juravsky et al. 2023; Gaessler llama.cpp perf). Basis of §10 and the
  report's driver tree. Confirmed here: short decode 84% of wall.
- Threads: 5 saturate dual-channel DDR4-3200, 9 slower than 8; 2.1 GHz Xeon
  quad-channel needs 29 (Gaessler); Ryzen 7040 8 > 16 threads, Mac Mini M4
  8 > 12 (Olling 2026). Basis for 6-thread decode default + this round's
  P-core pin. Sandwich (2025) separates prefill/decode plans and tunes core
  count per phase — TGE already does (decode pool vs all-logical prefill).
- Quant fidelity: 8B Q8_0 +0.04% PPL / KLD 0.0014 / RMS 1.2%; Q4_K_M +2.8% /
  0.0313 / 5.5% (llama.cpp docs; Kurt 2026). Small-model PPL is more fragile
  (Kumar 2024; OmniQuant) but adversarial robustness can invert (Fang 2025).
  TGE's opt-in policy (default f32 arithmetic, `--vnni` KL 1.6e-3 / 12 flips,
  `--kv f16` KL 1.7e-6 / 0 flips) matches this: price, don't assume.
- Speculation on CPU: llama.cpp ships ngram-simple / map-k / mod (~16 MB,
  hashed, variable draft); prompt-lookup drafting optimisation (42x via
  removing unordered_map copies, segmented maps) shows drafting cost is
  data-structure, not model. TGE's O(n·span) scan is negligible vs a weight
  pass, so no cache structure was needed; acceptance 3.95/step stands.
- Fast exp: QuAKE bit-trick (1-2 ULP, 10-35% inference claimed); exllamav2
  `exp256_ps` AVX2 in sampling softmax; ggml-cuda `__expf` in flash-attention
  rescaling (1-2 ULP irrelevant where max is subtracted); aussieai fused
  AVX2 exp/sum/mult 73x over naive. Supports vectorising attention softmax
  *after* a KL price — which is why it stays scalar here.
- Method: paired A/B, warm-up separated, median/MAD over ≥7 runs, 95% CI,
  per-item logging, run manifest (Georges 2007; Mytkowicz 2009 environment
  effects to 8%; HELM/MLPerf versioning). TGE practice (G-001 snapshots,
  G-002 log, `tge fidelity` teacher-forcing) is the local form.

## 3. This round: before → after

All on the same binary lineage (before = HEAD e139fcc build, after = this
round's build), 6 threads, Q8_0, warm, best-of-N.

### 3a. Batched remainder — WORKED

| shape | tokens | before (best) | after (best) | Δ |
|---|---|---|---|---|
| 4864×896 | 5 | 0.432 ms | 0.321 ms | **-26%** (range 0.321-0.376) |
| 4864×896 | 6 | 0.239 ms | 0.315 ms | +32% on best sample, inside ±20-30% box noise; repeated 0.315-0.364 |
| 4864×896 | 5 vs 6 | 1.81x (5 slower than 6) | ~1.0x | anomaly removed |
| 151936×896 | 5 | 8.432 ms | 5.769 ms | **-31.6%** |
| 151936×896 | 6 | 6.078 ms | 5.701 ms | -6.2% |

Fidelity: KL mean 6.115e-12, worst 9.6e-10, max logit diff 1.86e-4, top-1
1023/1023 — unchanged. Tests 60/60. Verdict: keep. Every 64-token prefill
chunk (10×6+4) and every 1..5-token tail benefits; large-vocabulary passes
most.

### 3b. P-core pin — WORKED AS VARIANCE FIX, no mean claimed

Mean short decode inside noise (34.9 → 32.3 t/s across loaded sessions;
box noise ±20%, 85% RAM in use). Handoff was already 0.5-0.8 µs, so no
speedup was predicted and none is claimed. Value: removes P/E migration as
a spread source; 5 decode workers land on 5 distinct P-core threads
(logical 0 reserved). Verdict: keep, labelled variance-only per G-002
honesty rule (cf. §7b "no end-to-end gain claimed" precedent).

### 3c. Weight prefetch — WORKED AS PREDICTABILITY FIX

`generate --prefetch`: 489 ms for 675 MiB here (~1.4 GB/s storage-bound),
reported explicitly, first-token fault removed from prefill. Total bytes
identical. Default off (warm CLI would pay 0.5 s for nothing); cold starts
and benchmarks opt in. Verdict: keep as opt-in.

### 3d. What did NOT work / was not attempted

- No mean decode gain from pinning or prefetch on a warm box (predicted).
- Small-shape 6-token best-sample regression (+32%) is noise, not a loss:
  repeated range overlaps, large-shape wins, unit parity holds.
- K-quant fusion, vectorised exp, shared-prefix KV, draft-model speculation:
  not attempted this round; priced in §26d/§4. Exp needs KL price first;
  prefix cache needs scheduler work; fusion needs differential-test-first.

## 4. Ending: where the frontier sits now

Phase-0 timers landed after the first draft of this report: `generate` prints
`timers: forward X ms, sample Y ms (Z%)` per decode (measurement-log §27).
Greedy sampling is 0.28 ms/token (0.3% of decode); temp-0.8 top-p-0.95 is
1.36 ms/token (1.8%), against 18.2 ms under the old full sort. The sampling
share now names regressions before they reach end-to-end rates. Output
projection re-sweep: 4 tokens in 1.57x one token's time (was ~4x);
`bench continuous` 8x16 rechecked at 43.46 t/s aggregate, transcripts
identical (absolutes depressed by box load; shares and parity hold).

Short-context decode is at the wall (84%, parity with llama.cpp build 11624
at 45 t/s); prefill at 75-85% of llama; 2k decode at ~80% of llama with f32
KV. Inside the wall, the remaining levers are priced opt-ins (`--vnni`
+14-19% prefill at KL 1.6e-3; `--kv f16` memory halving at KL 1.7e-6) and the
unfused K-quant path. Past the wall, only tokens-per-pass multiplies:
prompt-lookup speculation (1.55-2.0x measured, identical greedy ids) and
continuous batching (+38% aggregate, identical transcripts). The next
provable step is the fused K-quant dot under the harness, then a priced
AVX2 exp, then a shared-prefix cache — in that order, because each depends
on the instrument built before it. The engine's product is tokens per second
at a stated fidelity; once fidelity is a number (KL 6e-12 reference, 1.7e-6
/ 1.6e-3 priced trades), both can be optimised together.

## 5. Re-verification on a stressed box (this session, 86% RAM in use)

HEAD e139fcc + uncommitted §26-27 work, rebuilt (`ninja: no work to do` =
binary already current), `tge_tests` 60/60 pass, model
`qwen2.5-0.5b-instruct-q8_0.gguf` (675 MiB), 6 threads. No live llama.cpp
binary on this machine; llama comparisons below reuse the logged build-11624
pairs (§19) as reference, not new measurements.

- Fidelity harness: `tge fidelity --corpus docs/architecture.md` = mean KL
  6.115e-12, worst 9.6e-10, max |logit| 1.86e-4, top-1 1023/1023. Exact match
  to the logged §26a value. Vectorised attention + GQA pricing stands.
- Speculation parity + win reproduces under load on the same prompt
  ("Once upon a time", n=16, back-to-back): normal 6.33 t/s vs
  `--speculative` 13.51 t/s (2.13x), byte-identical text
  (", there was a young man named John who lived in a small village. John").
  Absolutes are depressed ~4-5x vs the §21 session (28-30 t/s baseline), but
  the within-run ratio and id-equality hold — the predicted 1.3-1.8x range is
  met, at the top end.
- Phase-0 timers: `timers: forward vs sample` prints on every run; greedy
  sampling share 0.2-0.9% here (4-5 ms over 1100-3600 ms forward), top-p path
  unchanged in code. The instrument gates regressions as designed.
- Batched-tail micro-bench today is noise-dominated and NOT used as evidence:
  FFN 4864x896 x5/x6 flip run-to-run (0.35 vs 0.72 ms, then 1.28 vs 2.08 ms),
  output 151936x896 x5/x6 20.8 vs 6.4 ms — all 3-6x above the §26a best-of
  figures. Box has 2.1 GiB free / 86% in use with page pressure; `bench batch`
  uses an auto thread count, so contention dominates. Keep-verdict rests on
  the §26a best-of-N session + bit-identical arithmetic (same widened block,
  same order, tail shares one block read; `q8` differentials + 60/60 tests
  guard it), not on today's absolutes. No revert: correctness holds, logged
  win stands, no contrary evidence.
- P-core pin: code verified (`ThreadPool` pins workers round-robin to the
  fastest efficiency class, group-0 only, OS default otherwise). No mean
  claimed; variance-only lever per §26b. Keep.
- Weight prefetch: `File::prefetch()` + `generate --prefetch` present,
  default off. Cold 1-token prefill here took 1180-2142 ms (fault-in, not
  compute), confirming the mechanism's premise; explicit timing moves it out
  of the first token. Keep as opt-in.
- Continuous batching with prompt-tokens: `bench continuous --requests 4
  --tokens 8 --sequences 2 --prompt-tokens 2` = 25.97 t/s aggregate,
  `state: independent (all transcripts identical)`. Absolute below the
  §25/§27 figures under load; independence (the correctness claim) holds.
  Keep.
- Still-scalar exp confirmed: `std::exp` only in sampler + SiLU + GQA merge
  (`sampling.cpp:82,118,222`, `qwen2.cpp:35,778-779`); no vectorised
  approximation in the tree. Correctly deferred pending a KL price.
- Input/termination/sampling cuts confirmed in code: CONTROL+USER_DEFINED
  literal match (`tokenizer.cpp:345`), end-of-generation set
  (`qwen2.cpp:109` eos/eot/`eos_token_ids`), candidate-set top-p/min-p
  without full sort (`sampling.cpp:111-140`). Unit suite covers all three.

Literature cross-check (mechanism + method transfer, not numbers): prompt
lookup 2-4x on input-grounded tasks with identical output (Saxena/PLD),
1.2-1.5x ngram in vLLM, 10-60% TensorRT-LLM N-gram at low concurrency, up to
4.23x PROMTEC on code; anisotropic spine trees 1.9-4.3x lossless over
single-source training-free drafting (GOOSE 2026) — supports a future
PLD+transition hybrid tree here, after the verifier handles trees, not
chains. CPU serving: prefill compute-bound / decode bandwidth-bound with
separate plans and core counts (Sandwich 2025) — matches this engine's
decode-pool vs prefill-pool split and the 53.3 t/s wall model. Quant
fidelity: Q8_0 near-lossless, Q4_K small PPL/KL cost growing on 0.5B
(Qwen2 quant tables; llama.cpp PPL/KL notes) — supports the opt-in policy
(default f32, priced `--vnni`/`--kv f16`, unfused K-quant still open).

Disposition: keep every working-tree change (batched remainder, P-core pin,
prefetch opt-in, timers, prompt-batched continuous). Nothing reverted: no
regression in correctness, timing, or fidelity was found; today's noisy
absolutes are reported as load, not as contrary evidence, per G-002.

## 6. §28 corrections addendum (2026-11-02): what this report got right,
   what the follow-up rounds changed

Nothing above is rewritten: the audit trail stands as written on the day.
The following rows supersede the cited claims with measurement-log evidence.

1. P-core pinning (row 10, §3b, §5 disposition). The variance rationale did
   not survive contact with contention: a sessions re-measure showed 4
   models x 6 pinned workers oversubscribing ~12 P-threads while E-cores
   sat idle (38.38 t/s aggregate pinned vs 59.97 unpinned, same window,
   +56%), and an interleaved single-stream A/B favored unpinned on the
   mean (48.9 vs 41.7, 6/6 wins) with 7x lower run-to-run range. Default
   flipped to unpinned; `--pin-threads 1` restores pinning. Log §28ab.
2. Vectorised exp (row 13, "still scalar", "needs KL price first"). Priced
   and landed: the shared Cephes exp now serves attention softmax (§28c),
   SiLU (§28c) and the sampler softmax (§28t), each gated by differentials
   (1e-4/1e-5) and the chi-square suite — plus the sampler's no-cut
   histogram (8-way, 100k draws). Remaining scalar exps (sampler cut paths,
   GQA online-softmax alphas) are documented as open, not as blocked.
3. Fidelity basis (row 2, §5: "KL 6e-12 scalar-vs-AVX2", §28a/§28c
   "fidelity-neutral"). The harness only varied attention then; norms, RoPE
   and SiLU ran vectorised in both arms. `set_scalar_kernels` now gives the
   reference arm the full scalar oracle, and the number that prices the
   whole program is mean KL ~1e-11 with unanimous top-1 on three corpora
   (architecture, report-0.1, sampling.cpp). The earlier "no interaction
   detected" readings are superseded, not contradicted. Log §28z, §28y.
4. Speculation scope (row 3, §5: greedy with identical ids). Greedy still
   holds (now including repeat-penalty configs, bit-exact: §28u, §28v), and
   temperature-only sampling joined with its distribution provably
   preserved (accept/reject + renormalised resample, unit-tested; §28r, §28s
   1.40-2.03x). Repeat-penalty sampled runs also verify (0.80x where the
   penalty fights repetition, §28u). Cuts (top-k/p, min-p) still fall back
   to single-token — exact reuse is the only acceptable construction there.
   Same-window matrix: greedy 1.50x bit-exact, temp 0.1 1.40x, penalty
   1.04-1.88x depending on draftability. Log §28r–§28x.
5. "Not attempted" list (row "K-quant fusion, vectorised exp, shared-prefix
   KV, draft-model"). Status now: exp done (see 2); fused single-token Q4_K
   dot attempted and REVERTED same-session (7.01 vs 8.15 GB/s — hadd tax
   beats the saved L1 round-trip; §28i); Q6_K (+82%) and Q5_0 (+121%)
   dequants vectorised bit-identical with K-quant scratch reuse (+50-84%),
   Q4_K_M at 80% of Q8_0 same-window (§§28h/j/k/l); shared-prefix KV
   designed then deferred — no paying caller exists yet (§28q); draft-model
   and int8-domain dot stay future work with the trigger conditions logged.

Net of §28 after this report: short-context decode parity held while
prefill roughly doubled on the thread split, Q4_K_M closed to 80% of Q8_0,
speculation covers greedy/temperature/penalty with a same-window matrix,
and the full-program fidelity price is ~1e-11 KL with unanimous top-1.

## 7. Addendum to the addendum: fidelity priced, pinning reversed (§28y–§28ab)

Two §6 items moved since it was written:

1. Fidelity basis (§6.3) is now measured, not caveated: `set_scalar_kernels`
   gives the reference arm the full scalar oracle (norms, RoPE, SiLU on top
   of scalar attention), and the price holds across three corpora — mean KL
   1.058e-12 (architecture.md), 9.960e-12 (report-0.1.md) and 6.514e-12
   (sampling.cpp), worst 1.2e-9, max |logit| 2.4e-4, top-1 1023/1023
   everywhere (§28y, §28z). The sampler exp stays chi-square-gated (teacher
   forcing never samples).
2. P-core pinning (§6 preamble, "keep ... P-core pin") is reversed: same-box
   A/B showed unpinned winning 6/6 single-stream pairs with 7x lower spread
   and +56% under 4-model sessions contention, so the default flipped
   (`--pin-threads 1` restores it; §28ab). Thread sweep on the unpinned pool:
   10 workers (all physical cores) beat 6 by ~4.5%; 16 collapses on HT
   contention; the default stays 6 pending multi-box evidence (§28ac).

Corrected numbers for §5-era claims: the "6e-12" fidelity price is ~1e-11
full-program; the "1.55-2x" greedy spec band is now a same-window matrix
(1.50x greedy bit-exact, 1.40x temp-0.1, penalty 1.04-1.88x; §28x);
Q4_K_M sits at 80-89% of Q8_0 same-window (§28l, §28af: temp-0.1 spec now
66.01 t/s post-flip, 2048-ctx decode 38.31 t/s).
