# INVENTED: VNNI i8 attention model wiring (stage 2) — KILLED (fidelity in class, but no reliable 8k decode gain)

- **Idea**: Wire the cycle-26 key-bsum VNNI QK dot into the model: store per-32 signed key sums with the i8 KV cache, pass them through `AttentionArgs`/`AttentionGroupArgs`, quantize the query once per head, and use the VNNI dot on AVX-VNNI CPUs for both the single-head and GQA group attention paths. Fall back to the existing float-query q8 dot otherwise.
- **Cycle date**: 2026-11-07
- **Verdict**: **KILLED as a model wiring.** Fidelity stays in the i8 class (off 2.272e-4 / 393/399; on 4.396e-4 / 394/399; ids identical on vs off), but the 8192-context decode A/B shows no reliable gain. The raw decode ratios (1.51x and 1.10x) track the **matvec** ratio in the same arms (1.55x and 1.07x) — a phase the VNNI dot never touches. Normalizing attention by matvec flips sign across the two pairs (1.061x and 0.927x). The QK dot is a minority of the attention phase, so a 1.61x dot dilutes to a few percent of decode, below the 1.15x wiring gate and inside this box's drift.
- **Tree state**: spike reverted from all six touched files; 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Mechanism

Stage 2 of the cycle-24/26 long-context direction:

1. `store_kv_q8` also writes `k_sums[block] = sum(signed key[32])`, parallel to the per-32 key scales.
2. `AttentionArgs` and `AttentionGroupArgs` gain `keys_q8_sums`.
3. `attention_head_avx2` quantizes the head query once to unsigned q8 + per-32 scales.
4. `attention_group_avx2` does the same per query head inside the group loop.
5. When `keys_q8_sums != nullptr`, `head_dim % 32 == 0` and AVX-VNNI is present, the QK score uses `attention_q8_dot_vnni_keysums`; otherwise the existing `q8_dot_avx2` path runs unchanged.
6. `TGE_VNNI_ATTN=0` (cached once) disables the VNNI branch for a same-binary A/B.

The value accumulation, softmax and all other tiers are untouched.

## 2. Falsifier

- **KEEP** only if 399-position i8 fidelity stays under the 1.6e-3 VNNI-class bar, ids are identical where the tier allows, and the 8192-context i8 decode is **≥1.15x** the current path in a drift-controlled pair.
- **KILL** if the gain is inside the box drift or below the gate.

Measured: fidelity passes, speed gate fails.

## 3. Measured numbers

### 399-position i8 fidelity (context 1024)

| arm | mean KL | worst KL | top-1 |
|---|---:|---:|---:|
| VNNI off | 2.272e-4 | 5.921e-3 | 393/399 |
| VNNI on | **4.396e-4** | 6.002e-3 | **394/399** |

Both are inside the i8 class and the 1.6e-3 VNNI-class bar. Query quantization roughly doubles the mean KL (2.27e-4 → 4.40e-4) but does not cost top-1 in this corpus.

### 512-token i8 id parity

Generated ids are **IDENTICAL** on vs off (`--n 32 --print-ids`). Decode read 2.22 vs 2.73 t/s, but at 512 tokens attention is a negligible share, so that difference is box drift, not VNNI.

### 8192-context A/B (8180-token prompt, 7 decode tokens, `--kv i8`)

Pair 1 — OFF then ON:

| phase | OFF | ON | ratio |
|---|---:|---:|---:|
| prefill | 627,683 ms | 747,073 ms | 0.84x |
| decode total | 6,826.3 ms | 4,523.9 ms | 1.51x |
| decode matvec | 3,951.5 ms | 2,548.1 ms | **1.55x** |
| decode attention | 2,820.2 ms | 1,930.3 ms | 1.46x |
| attention / matvec | 0.714 | 0.757 | 1.061x |

Pair 2 — ON then OFF:

| phase | ON | OFF | ratio |
|---|---:|---:|---:|
| prefill | 771,845 ms | 861,137 ms | 1.12x |
| decode total | 4,646.5 ms | 5,114.6 ms | 1.10x |
| decode matvec | 2,574.0 ms | 2,746.1 ms | **1.07x** |
| decode attention | 2,026.5 ms | 2,331.7 ms | 1.15x |
| attention / matvec | 0.787 | 0.849 | 0.927x |

Reading:

- The arm with faster decode also has proportionally faster **matvec**, which the VNNI branch cannot change. The whole raw "gain" is box drift between arms.
- The drift-normalized attention ratio moves **+6% in one pair and −7% in the other** — no detectable signal.
- Prefill also drifts by 0.84x/1.12x between arms, confirming the window was not stable.

## 4. Why it lost

- **The QK dot is a minority of attention.** The model attention phase also performs the softmax pass, the weighted value accumulation over the same span, and the i8 widen/scale work. Cycle 24 estimated QK at roughly half of attention; the model-level A/B is consistent with a much smaller share.
- **1.61x on a minority phase cannot clear 1.15x decode.** Even at a 35% QK share a 1.61x dot saves ~13% of attention, and attention is 41–46% of decode here — about 5–6% decode, before drift.
- **The measurement is below the box's resolution.** This window's arm-to-arm drift is ±30–50% in matvec, larger than the expected 3–6% effect; a same-order pair cannot resolve it in the time budget, and a full ABBA campaign at ~13 min prefill per arm is not justified for a few percent.

## 5. Disposition

- Do **not** wire the VNNI QK dot into the model. The cycle-26 kernel result (1.61x median on an 8192-key span, 0.28% RMS error) remains a report-only kernel finding; it is not a model win.
- Keep the current float-query q8 attention for the i8 KV tier.
- The 8k reopen trigger from cycle 24 is now priced end-to-end: the attention **phase** is large, but the QK **dot** inside it is not, so a QK-only kernel is the wrong lever at this share.
- If the long-context attention line is revisited, target the **value accumulation** and softmax passes (or a fused attention micro-kernel), not the QK dot, and measure device-level with an interleaved ABBA that can resolve a few percent.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
# fidelity
TGE_VNNI_ATTN=0 tge fidelity --model Llama-3B --corpus docs/architecture.md \
  --context 1024 --tokens 400 --threads 6 --gpu off --kv i8
tge fidelity ... (same, VNNI on)
# id parity
TGE_VNNI_ATTN=0 tge generate --model Llama-3B --tokens <512 ids> --n 32 \
  --threads 6 --gpu off --kv i8 --print-ids
# 8k A/B (temporary @file token helper)
TGE_VNNI_ATTN=0 tge generate --model Llama-3B --tokens @ids8180.txt --n 8 \
  --threads 6 --gpu off --kv i8 --context 8192
tge generate ... (same, VNNI on)
git checkout -- include/tge/cpu/kernels.hpp include/tge/models/qwen2.hpp \
  src/models/qwen2.cpp src/cpu/avx2/q8_batch_vnni.cpp \
  src/cpu/avx2/attention_avx2.cpp apps/cli/main.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
