# MEASURED: speculation at the 8k boundary on 3B — still 1.45-1.61x; attention reaches 52-56% of decode, and the 4k compression is not a law

- **Idea**: The speculative context table had 2k (2.74x) and 4k (2.02x) points on the fox workload, but no point at the 8k boundary. The verifier reads six query rows against the full KV, so at 8k the attention term should dominate the step cost; the risk was that speculation becomes a net loss near the maximum tested context, which would require a context cutoff in the long-prompt auto-spec policy. Measure the missing endpoint.
- **Cycle date**: 2026-11-12
- **Verdict**: **MEASURED — speculation still pays at 8k.** Llama-3.2-3B-Instruct-Q4_K_M, 8,193-token periodic prompt, `--context 8704 --n 16 --kv f16 --threads 10 --gpu off`:
  - off: **5.21 / 5.68 t/s**; decode phases `matvec 43-45%, attn 54-56%, norm 1%`.
  - spec: **8.22 / 8.38 / 8.09 t/s**; `4 steps, 12 draft tokens accepted (3.00/step), 16 emitted`; phases `matvec 45%, attn 52%, norm 3%`.
  - ratios: 1.42-1.61 across the six off/spec pairings; **median 1.51x**.
  - The six-row verify amortizes the KV read: per-row attention is 39-40 ms versus 94-106 ms for a single-row decode token, so step cost grows well below 6x in rows even when attention is over half of decode.
  - A controlled 4k point on the same compact prompt lands at **1.24-1.29x** (interleaved pairs), so the 2k 2.74x -> 4k 2.02x drop on the fox workload is not a universal context law; the ratio is workload- and box-state-dependent, and spec remains positive at every measured context.
- **Tree state at close**: no code touched; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Llama-3.2-3B-Instruct-Q4_K_M.gguf |
| prompt | `The fox runs. ` x 2048 = 8,193 tokens |
| context / n / threads / KV | 8704 / 16 / 10 / f16 |
| arms | off vs `--speculative` |
| runs | off 2, spec 3 at 8k; off 2, spec 2 interleaved at 4k |

**Why a compact sentence.** The 2k/4k points used `The quick brown fox jumps over the lazy dog. ` x 410 (4,100 tokens, ~18.5k characters). At 8k that text is ~37k characters, over the Windows 32,767-character CreateProcess limit, and the CLI has no prompt-file or stdin input. `The fox runs. ` tokenizes to 4 tokens in 14 characters, so 2,048 repetitions give 8,193 tokens in ~28.7k characters and pass through `Start-Process` as one quoted argument. The trade is a 4-token period (3.00 accepted/step at 8k, 4.00 at 4k) instead of the fox 10-token period (5.33/step at 4k); the 8k point is therefore not a drop-in continuation of the fox curve, which is why a controlled 4k point on the same sentence was measured in this cycle.

## 2. Results at 8k

| run | off t/s | spec t/s | ratio | spec line | decode phases (off / spec) |
|---|---:|---:|---:|---|---|
| a | 5.21 | 8.22 | 1.58 | 4 steps, 12 accepted (3.00/step), 16 emitted | off matvec 1236.0 ms (43%), attn 1597.5 ms (56%), norm 38.7 ms (1%) / spec matvec 818.9 ms (45%), attn 945.8 ms (52%), norm 51.5 ms (3%) |
| b | 5.68 | 8.38 | 1.48 | 4 steps, 12 accepted (3.00/step), 16 emitted | off matvec 1188.1 ms (45%), attn 1409.9 ms (54%), norm 35.5 ms (1%) / spec matvec 799.2 ms (45%), attn 931.0 ms (52%), norm 50.8 ms (3%) |
| c | 5.68 | 8.09 | 1.42 | 4 steps, 12 accepted (3.00/step), 16 emitted | (off reused) / spec matvec 828.5 ms (45%), attn 967.6 ms (52%), norm 48.7 ms (3%) |
| **median** | **5.45** | **8.22** | **1.51** | | |

Cross pairings span **1.42-1.61x** (8.38/5.21 = 1.61 is the high end, 8.09/5.68 = 1.42 the low end). The spec arm ran after the off arm and still won, so the 1.4x floor is not an ordering artifact; the runs were sequential rather than interleaved because each 8k prefill costs ~230-240 s.

Spec run c with `--print-ids` continued the periodic sentence exactly:

```text
ids      : 578 39935 8640 13 578 39935 8640 13 578 39935 8640 13 578 39935 8640 13
```

(578 = `The`, 39935 = ` fox`, 8640 = ` runs`, 13 = `.`.) No off-arm id capture was taken this cycle, so no parity claim is made here; greedy spec is verifier-exact by construction (ADR-002) and the 2k/4k fox tables recorded identical ids.

## 3. Controlled 4k point on the same prompt

`The fox runs. ` x 1024 = 4,097 tokens, `--context 4608`, same arms. Interleaved off/spec pairs (off, spec, spec, off):

| pair | off t/s | spec t/s | ratio | spec line |
|---|---:|---:|---:|---|
| 1 | 7.33 | 9.07 | **1.24** | 4 steps, 16 accepted (4.00/step), 19 emitted |
| 2 | 6.61 | 8.51 | **1.29** | 4 steps, 16 accepted (4.00/step), 19 emitted |

An earlier sequential arm order (off, off, spec, spec) read 8.98/8.49 off versus 7.81/7.85 spec (0.87-0.92), but that control is **inadmissible**: the context-independent matvec phase in the spec arm was 260-303 ms per six-row step versus 200-207 ms in the 8k spec runs (1.3-1.5x slower), i.e. the later spec arm ran on a degraded box state, not a slower algorithm. The interleaved pairs cancel that drift; the 4k compact ratio is ~1.26x.

## 4. Reading

- **Speculation pays at 8k**: even with attention over half of decode, the six-row verify shares each KV row across six queries. Per-row attention is 236-242 ms / 6 = 39-40 ms at 8k, against 94-106 ms per single-row token in the off arm (56% of decode). The step cost therefore grows sublinearly in rows, and four steps still finish 16 tokens faster than sixteen single-token forwards.
- **The 2k -> 4k fox compression is not a context law.** The fox table (2.74x at 2k, 2.02x at 4k) suggested ~1.5x at 8k by linear attention growth. The compact-prompt control shows a different answer: ~1.26x at 4k and ~1.5x at 8k on the same sentence, with box drift comparable to the difference. The ratio depends on accepted tokens per step and on how well the batch attention amortizes the KV read, not on context alone.
- **No context cutoff is warranted** for `--speculative` or for the long-prompt auto gate on this evidence: the minimum measured ratio is 1.42x at the maximum tested context, and the spec arm's runs were the later (warmer) ones. A cutoff would give up a real win.
- **Usage guidance**: keep `--speculative` on repetitive prompts at 8k; expect **>=1.4x**, not the 2-3x seen at short context, and read the phase split (attn 52-56%) as the reason. At longer context than 8k the KV tiers/sparse levers remain the first-order tools; speculation composes on top rather than replacing them.
- No fidelity run owed: measurement only, no arithmetic change. The spec arm pins per-head attention as shipped (ADR-002).

## 5. Falsifier (stated before the runs)

**KEEP/MEASURED** if the 8k spec ratio is >= 1.2x with a plausible accept rate and no output anomaly. **KILL** if the ratio is <= 1.0x or the step count balloons (e.g. drafting collapse). Measured 1.42-1.61x with 3.00 accepted/step: keep the lever, no code change.

## 6. File anchors (current HEAD, read-before-write verified)

- `apps/cli/main.cpp:788-922` — fused greedy speculative chain (`feed[0]` = bonus, five span drafts, six-row verify).
- `apps/cli/main.cpp:690-694` — prompt + generated vs `model->max_context()` fit check; `:678` prints the prompt token count before the check, which is how the 8,193-token prompt was confirmed with a 64-token context.
- `include/tge/models/qwen2.hpp` — `kSpeculativeBatch = 6`.
- `reports/idea-invented-spec-at-4k-measured.md`, `reports/idea-invented-spec-at-2k-measured.md` — the earlier curve points this cycle extends.
- ADR-002 (`.aikit/decisions.md`) — speculation pins the per-head attention schedule.

## 7. Reimplementation recipe

1. Build the compact prompt in PowerShell (the fox text does not fit the Windows command line at 8k):

   ```powershell
   $p = ("The fox runs. " * 2048)          # 8,193 tokens, ~28.7k chars
   ```

   Confirm the token count cheaply by passing a tiny context; the CLI prints `prompt : N tokens from text` before the fit error:

   ```powershell
   tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt $p --context 64 --n 1 --threads 10 --gpu off
   ```

2. Run both arms with `Start-Process` and redirected streams (a 28k-character prompt plus `2>&1 | Out-File` truncates in the PowerShell pipeline):

   ```powershell
   $a = "generate --model `"$model`" --prompt `"$p`" --context 8704 --n 16 --kv f16 --threads 10 --gpu off[ --speculative]"
   Start-Process -FilePath $exe -ArgumentList $a -RedirectStandardOutput $out -RedirectStandardError "$out.err" -NoNewWindow -Wait
   ```

3. Expect off ~5.2-5.7 t/s (attn 54-56%), spec ~8.1-8.4 t/s with 4 steps and 3.00 accepted/step, ratio 1.42-1.61x. Each 8k run costs ~230-240 s of prefill on this box.
4. For the controlled 4k point, use `("The fox runs. " * 1024)`, `--context 4608`, and interleave the arms; expect ~1.24-1.29x.
5. No code ships.

## 8. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56)
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed
tge generate ... --context 64  (token count)      -> prompt: 8193 tokens
tge generate ... --context 8704 --n 16 --kv f16 --threads 10 --gpu off
  -> 5.21 / 5.68 t/s; attn 54-56% of decode
tge generate ... --speculative
  -> 8.22 / 8.38 / 8.09 t/s; 4 steps, 12 accepted (3.00/step), 16 emitted; attn 52%
tge generate ... --context 4608 (compact 4k, interleaved)
  -> off 7.33 / 6.61, spec 9.07 / 8.51; ratios 1.24 / 1.29
git status --short                                -> clean (no code touched)
python aikit.py check aegean-tge                  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
