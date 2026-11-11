# INVENTED: whole-FFN kq8d (`ffn,down`, gate+up+down integer) — KILLED on the fidelity gate by exactly one top-1 flip (97.49% vs 97.6%), despite +8% over `ffn`

- **Idea**: The `ffn` scope (kq8d on gate+up) is a passing tier at 2.495e-4 / 198/199 and ~1.20x decode; the re-priced `down` tier is also positive; the whole SwiGLU FFN is 56% of the per-token weight stream. Route gate, up **and** down through the integer row kernel (`--kq8d ffn,down`) and price the stack. Predicted: 1.25-1.35x decode, KL under 1.6e-3, top-1 held.
- **Cycle date**: 2026-11-11
- **Verdict**: **KILLED on the pre-stated fidelity gate — by one position.** Fidelity at 399 positions (context 1024, f32 KV): mean KL **9.100e-4** (inside the 1.6e-3 class bar), worst 7.940e-3, top-1 **389/399 = 97.49%** — the bar is >= 97.6%, i.e. >= 390/399. The 199-position run agrees: 9.609e-4, **194/199 = 97.49%**, one flip under the >= 195 needed. The speed case is real and would have shipped: `ffn,down` is **1.250/1.321x** over off (vs `ffn` 1.181/1.198x) and adds **1.059/1.102x** on top of `ffn` in the same pairs, ids identical on the test prompt. But a tier that crosses the class bar cannot join the passing set; `ffn` alone stays the recommended FFN tier.
- **Tree state at close**: spike reverted from `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp`, `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

Extends the `ffn` scope (`reports/idea-invented-kq8d-ffn-keep.md`) to the down projection:

- `Qwen2::set_kq8d(head, down, attn, gateup)` already has all four flags after the `ffn` spike; the combined tier simply sets `gateup = true` **and** `down = true`.
- Decode calls: `matvec(layer.w_gate, ..., kq8d_gateup_)`, `matvec(layer.w_up, ..., kq8d_gateup_)`, `matvec(layer.w_down, ..., kq8d_down_)`; all three then dispatch to `gemv_dequant_kq8_row_avx2` (activation quantised once per call into a `thread_local` Q8_K scratch; rows banded over the decode pool).
- CLI: `--kq8d ffn,down` (and `down,ffn`) sets both; usage text extended.
- Everything else (q/k/v/o, LM head, attention, KV) stays on the fused-float path.

## 2. Predicted gain (with the math)

Weight shares of the 2.02 GiB per-token stream: gate+up ~756 MiB (37%), down ~378 MiB (19%). `ffn` measured 1.20x on the gate+up share, i.e. an effective ~1.5x on that share (1/(1-0.37+0.37/1.5)=1.18). Down alone measured 1.07-1.22x, i.e. ~1.3-2x effective on its share. Multiplying the passes predicts `1.20 x 1.07 ≈ 1.28x`; the measured 1.25-1.32x matches. The fidelity arithmetic is the risk: gate+up alone is 2.495e-4 (99.50% top-1) and down alone is 7.954e-4 (97.74% at 399 positions). Independent errors would put the stack near 1.0e-3 — measured 9.1e-4 — but the top-1 flips concentrate at the same near-tie positions the down tier already sits on, so the stack lands one position below the class bar rather than mid-class.

## 3. Falsifier (stated before the spike)

**KEEP as opt-in** if 399-position f32 fidelity has mean KL <= 1.6e-3 **and** top-1 >= 97.6% (>= 390/399) **and** the decode median over `ffn` is >= 1.05x with ids consistent. **KILL at the first gate that fails**, no further promotion.

Fidelity failed at 97.49% (389/399). Kill, even though the speed gate passed.

## 4. File anchors (current HEAD, read-before-write verified)

- `include/tge/models/qwen2.hpp:241-259` — `set_kq8d` + scope getters (spike adds `gateup`/`kq8d_gateup_`).
- `src/models/qwen2.cpp:2228-2229` — gate/up matvec calls; `:2233-2234` — down matvec call.
- `src/models/qwen2.cpp:969-1020` — `matvec` kq8d-first dispatch.
- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:397-456` — `gemv_dequant_kq8_row_avx2`.
- `apps/cli/main.cpp:2286-2306`/`:2805-2825` — `--kq8d` parsers; `:506`/`:2486` — command signatures; `:591`/`:2549` — `set_kq8d` call sites.
- `reports/idea-invented-kq8d-ffn-keep.md` — the passing `ffn` tier and its fidelity (2.495e-4 / 198/199).
- `reports/idea-invented-kq8d-tiers-repriced.md` — down/head tier re-pricing and the rotation lesson.
- `reports/idea-invented-kq8d-down-keep.md` — down fidelity 7.954e-4 / 390/399.

## 5. What was done (temporary spike, fully reverted)

1. Re-applied the `ffn` wiring: fourth `set_kq8d` parameter + `kq8d_gateup_`, gate/up matvec flags, CLI plumbing through `command_generate`/`command_fidelity`, `ffn` parse case.
2. Added the combined scope: parse `ffn,down` / `down,ffn` to set `kq8d_gateup = true; kq8d_down = true;`; usage text `head|down|attn|ffn|ffn,down|head,down`.
3. Built, 171/171. Fidelity runs at 199 positions (context 512) and 399 positions (context 1024), threads 1, f32 KV, docs/architecture.md.
4. Decode A/B: same binary, rotated orders (off,ffn,ffn,down then ffn,down,ffn,off), dragon prompt, `--n 48 --kv f32 --threads 10 --gpu off --print-ids`.
5. Reverted all three files; rebuilt; 171/171; gates 4/4; clean tree.

## 6. Measured numbers

### Fidelity (f32 KV, scalar oracle, docs/architecture.md)

| positions | mean KL | worst KL | top-1 | bar | result |
|---:|---:|---:|---:|---|---|
| 199 (ctx 512) | 9.609e-4 | 7.940e-3 | 194/199 (97.49%) | >= 97.6% | **fail by 1 flip** |
| 399 (ctx 1024) | 9.100e-4 | 7.940e-3 | 389/399 (97.49%) | >= 390/399 | **fail by 1 flip** |

For comparison, the passing tiers: `ffn` 2.495e-4 / 198/199 (99.50%); `down` 7.954e-4 / 390/399 (97.74%); `head,down` 9.711e-4 / 390/399 (97.74%).

### Decode (dragon prompt, n=48, f32, threads 10, rotated order)

| pair | off t/s | ffn t/s | ffn,down t/s | ffn/off | stack/off | stack/ffn | ids |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 16.95 | 20.01 | 21.19 | 1.181 | 1.250 | 1.059 | identical |
| 2 | 16.38 | 19.63 | 21.63 | 1.198 | 1.321 | 1.102 | identical |

The adding-down increment (~1.08x median) is consistent with the re-priced down tier; the stack is ~1.28x over off.

## 7. Why the kill is the right call

- The class bar exists to keep near-tie risk bounded; a stack at 97.49% is **below** both the `head,down` precedent (97.74%) and the floor (97.6%). One position on 399 is small, but the rule is the rule, and two independent sample sizes landed at the same 97.49% — this is not sampling noise around a passing value, it is a stable boundary crossing.
- The speed is not the problem: +8% over `ffn` is real and would be attractive. If a future cycle finds a cheaper activation-quantisation scheme (per-32 symmetric scales, the round-4 prescription) that lifts the stack to >= 97.6%, re-run this exact recipe — the decode and fidelity harnesses are already specified above.
- Until then the passing recommendation is `--kq8d ffn` (1.20x, 99.50% top-1) and, separately, `--kq8d down` (down-only, 97.74%). Do not stack them.

## 8. Full reimplementation recipe

Do not ship `ffn,down`. To reproduce the kill:

1. Apply the `ffn` wiring from `idea-invented-kq8d-ffn-keep.md` §8 (fourth `set_kq8d` flag `gateup`, `kq8d_gateup_`, gate/up matvec flags, CLI plumbing, `ffn` parse case).
2. In both `--kq8d` parsers add: `else if (value == "ffn,down" || value == "down,ffn") { kq8d_gateup = true; kq8d_down = true; }`; extend the error text and both usage blocks to `head|down|attn|ffn|ffn,down|head,down`.
3. Build + `tge_tests` 171/171.
4. `tge fidelity --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --corpus docs/architecture.md --context 1024 --tokens 400 --threads 1 --gpu off --kq8d ffn,down` -> expect mean KL ~9.1e-4, top-1 389/399 (fail). Repeat at `--context 512 --tokens 200` -> 194/199 (fail).
5. Decode: `tge generate --n 48 --kv f32 --threads 10 --gpu off --print-ids [--kq8d ffn|ffn,down]` with rotated arm order -> expect ffn/off 1.18-1.20, stack/off 1.25-1.32, stack/ffn 1.06-1.10.
6. Revert; the finding is the fidelity number, not the speed.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge fidelity ... --context 512  --tokens 200 --threads 1 --gpu off --kq8d ffn,down
  -> mean KL 9.609e-4, worst 7.940e-3, top-1 194/199 (one short)
tge fidelity ... --context 1024 --tokens 400 --threads 1 --gpu off --kq8d ffn,down
  -> mean KL 9.100e-4, worst 7.940e-3, top-1 389/399 (one short)
tge generate ... --n 48 --kv f32 --threads 10 --gpu off --print-ids [--kq8d ffn|ffn,down]
  -> ffn/off 1.181/1.198; stack/off 1.250/1.321; stack/ffn 1.059/1.102; ids identical
git checkout -- include/tge/models/qwen2.hpp src/models/qwen2.cpp apps/cli/main.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the rebuilt binary; the committed tree is unchanged except this report.
