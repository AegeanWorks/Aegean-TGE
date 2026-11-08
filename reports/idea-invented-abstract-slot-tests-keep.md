# INVENTED: abstract KV-slot property tests with a deterministic nonzero fixture — KEEP-GRADE (tests pass; both mutations caught)

- **Idea**: The existing synthetic phi fixture has zero weights, so every logit is zero and cross-slot KV corruption is invisible. Build a deterministic **nonzero** one-block phi model, then assert two abstract properties: (1) slot 1's history must be bit-identical to a fresh single-slot model with the same history, and (2) resetting slot 1 must leave slot 0's replay bit-identical while slot 1 replays like a fresh slot. Mutation-test both properties to prove they are not vacuous.
- **Cycle date**: 2026-11-08
- **Verdict**: **KEEP-GRADE test recipe.** The two properties pass on today's tree (164/164), and both deliberately broken implementations are caught. The first version of the slot test was itself blind and was strengthened after mutation 1 exposed it. Per the loop protocol the spike is reverted; the fixture and tests below are ready to integrate as permanent coverage.
- **Tree state**: spike reverted from `tests/test_gguf.cpp`; branch `idea/abstract-slot-tests` deleted; `git status` clean; master rebuilt 162/162; nothing committed except this report.

## 1. Why the old fixture cannot see the bug

`build_phi_model()` writes zero bytes for every weight, a zero scale in every Q8_0 block, and zero F32 vectors. The forward pass therefore produces all-zero logits for every input, so:

- a read from the wrong KV slot,
- a write that overwrites another slot,
- a reset that clears the wrong range,
- a stale CoW row,

all produce the same zero logits. The old forward smoke test only verifies finiteness and determinism, not state isolation.

## 2. Deterministic nonzero fixture

Extend `build_phi_model(bool patterned = false)` (same shape: 1 block, dim 32, 2 query heads, 1 KV head, head_dim 16, kv_dim 16, 64-token vocab, 10 tensors). When `patterned`:

- Q8_0 tensors: each 34-byte block gets fp16 scale bytes `0x00, 0x2C` (0x2C00 = 0.0625, finite and small), then 32 signed quants from an LCG: `state = state*1664525 + 1013904223; value = ((state >> 24) % 17) - 8`.
- F32 tensors: norms `0.5 + 0.05*(i % 7)` (positive, non-degenerate); `rope_factors_short` all `1.0`.
- Each tensor's data is appended in entry order with the same 32-byte alignment as the offsets, so the tensor table stays consistent.

This produces finite, input-dependent logits while keeping the file tiny (a few KiB).

## 3. Properties and results (164/164 pass)

### P1 — `model_sequence_slots_are_independent`

- Load with `sequences=2`.
- slot 0: prefill prompt `P = {1,2,3,4}`, decode `A = {9,10,11,12}` at positions 4-7; snapshot logits.
- slot 1: prefill the same `P` (exercises the global CoW hit path), decode `B = {13,14,15,16}`; snapshot.
- fresh `sequences=1` model: prefill `P`, decode `A`; compare slot 0 to fresh bit-for-bit. Then `reset_cache()`, prefill `P`, decode `B`; compare slot 1 to fresh bit-for-bit.

Pass: slot 0, slot 1 and the fresh model agree exactly, so slot offsets, per-slot writes, and CoW memcpy rows are all bit-identical.

**Test-design lesson**: the first version decoded the same tokens in both slots. Under the "force attention to slot 0" mutation it still passed, because both slots held identical histories — a wrong-slot read was unobservable. Giving slot 1 a different decode history fixed the blindness.

### P2 — `model_reset_one_slot_leaves_other_slots_intact`

- slot 0: prefill `{1,2,3,4}`, decode `{9,10,11,12}` to position 7; snapshot the position-7 logits.
- slot 1: prefill `{5,6,7,8}`, decode `{13,14,15,16}`.
- `reset_cache(1)`, then replay `forward(0, 12, 7)`; the logits must equal the snapshot bit-for-bit.
- Re-prefill slot 1 and decode `{13,14,15,16}`; compare to a fresh single-slot model with the same history.

Pass: resetting slot 1 leaves slot 0 untouched, and slot 1 behaves like a fresh slot.

## 4. Mutation evidence (the tests have power)

| Mutation | slot test | reset test |
|---|---|---|
| `attention(sequence, ...)` -> `attention(0, ...)` in `forward` | **FAIL** | **FAIL** |
| `reset_cache(sequence)` begin forced to 0 (clears slot 0) | pass (expected) | **FAIL** |

Each mutation was built and run, then reverted. The first slot-test version passed mutation 1; after differentiating the slot-1 history it fails, which is the point of the mutation check.

## 5. Full reimplementation recipe

1. `tests/test_gguf.cpp`, `build_phi_model(bool patterned = false)`: after the tensor table and header padding, if `patterned`, append each entry's data with the scale/quant/float pattern above instead of zeros; keep the `(bytes + 31)/32*32` alignment.
2. Add `model_sequence_slots_are_independent` and `model_reset_one_slot_leaves_other_slots_intact` after the existing `model_rejects_i8_kv_with_unaligned_kv_dim` test, using the exact histories above and `std::memcpy` snapshots of `vocab` floats.
3. Compare with `==` (bitwise), not a tolerance; the property is exact equality to a fresh model.
4. Run `tge_tests` (expect 164/164 with the two new cases).
5. Mutation check (optional but recommended once): force `attention(0, ...)` and confirm both tests fail; force the reset range to start at 0 and confirm the reset test fails; revert.

## 6. Verdict and follow-up

**KEEP-GRADE**: the tests are deterministic, fast (a few milliseconds), and catch two real state bugs. They should be integrated as permanent coverage in a normal (non-spike) change; the loop keeps this report only.

Next abstract-test ideas in the same family:

- A **position-gap** property: after a reset, a forward at a position whose earlier rows were overwritten by a *shorter* history must match a fresh model with the same shorter history; this catches a reset that leaves stale rows beyond the new prefix.
- A **CoW tier-tag** property: a hit on an f32 table must not be served to an f16/i8 cache (the table is f32-only by construction; a negative test could assert no hit).
- A **mutation harness** that breaks each slot-index/stride expression once and asserts the property test fails; this turns the one-off mutation check into a reusable gate.
