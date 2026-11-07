# INVENTED (gpu-prefill follow-up): GPU Offload Triage — CORRECTED (weights ARE cached; the dominant cost is uncached output readback ~72%, not re-uploads)

- **Idea**: My REOPEN report blamed non-persistent weights for the 12.9 s offload (4.3 GB moved vs ~0.5 GB model). Triage it properly with code reading + the built-in `TGE_GPU_TRACE` per-phase timer: find whether weights re-upload (design gap / shadowing bug) or something else dominates.
- **Cycle date**: 2026-11-06
- **Verdict**: CORRECTED with a precise replacement. (1) Weights ARE cached: `ensure_resident` keys full-matrix (pointer, bytes); h2d p50 = 0.000 ms across 5216 calls — first-touch uploads only (max 90.8 ms), then free hits. (2) The 4.3 GB "moved" is activation I/O (the byte counter sums per-call in+out, not re-uploads) — my smoking gun was misread accounting, retracted here explicitly. (3) The real dominant cost is the d2h output readback: 9557.9 ms of 13.2 s traced (72%), consistent with CPU reads from write-combined host-visible memory (~1-2 ms per call including small rows). The fix direction is named, not implemented (human gate; needs Vulkan validation).
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). No branch, no edits (read-only triage + existing trace flag), scratch deleted. `master` ahead of `origin/master` by 116 after this report. Nothing pushed (human gate).

## 1. Mechanism (as found — no code changed)

- Upload cache: `src/gpu/vulkan_context.cpp:44-57` (`g_residents`, one device-local buffer per exact (pointer, bytes)); `ensure_resident` (:799-910) returns resident + `h2d_ns=0` on hit; callers (`gemv_q8` :1099, `gemm_q8` :1199) pass full-matrix base+bytes, so keys are stable across calls.
- Byte counter: `qwen2.cpp:930-933` accumulates `tokens × (hidden + logits bytes)` per offloaded call — activation I/O accounting, NOT weight traffic.
- Per-call path `run_q8_call` (:983-1074): persistent mapped I/O (`ensure_io` grows only), descriptor alloc + command recording (`alloc_ns`), single submit + full fence wait (`submit_ns`, includes kernel time), output `memcpy` (`d2h_ns`).
- I/O buffer memory type (`grow_io_buffer` ~:555-594): first HOST_VISIBLE|HOST_COHERENT type, NO HOST_CACHED preference → write-combined on discrete GPUs: ideal for CPU writes (inputs), pathological for CPU reads (outputs).

## 2. Predicted gain — triage, landed with a correction

Falsifier: find either (a) no caching / per-call weight upload (design gap confirmed as filed) or (b) a working cache (headline wrong, find the real cost). Measured (b): first touches then zeros, ~72% in d2h.

## 3. Falsifier outcome — headline corrected, mechanism named

1. **Weights persist** (p50 h2d = 0.000 ms over 5216 calls; total h2d 438 ms ≈ first-touch sum). The REOPEN report's mechanism is withdrawn; its NUMBERS stand (offload still loses 2.5x), only the blamed cause changes.
2. **Dominant cost is output readback**: d2h 9557.9 ms total (mean 1.8, p50 1.28, max 9.4 ms/call) vs submit 3004.7 / h2d 438.3 / alloc 224.0. Consistent with uncached WC reads (even 3.5 KB outputs cost ~ms: PCIe round-trips per line).
3. **submit_ns (3.0 s) is kernel+sync bundled** — cannot split without GPU timestamp queries (named as the remaining unknown, not guessed at).

## 4. File anchors (all read, none touched)

- `src/gpu/vulkan_context.cpp:44-57, :799-910` (upload cache + ensure path).
- `src/gpu/vulkan_context.cpp:983-1074` (per-call path; `TGE_GPU_TRACE` split used for measurement).
- `src/models/qwen2.cpp:930-933` (the byte counter — what "moved" actually counts).
- Grounds: gpu-validate (correctness baseline, stands) + gpu-prefill-reopen (numbers stand, mechanism corrected here).

## 5. What was done (read-only triage + existing trace flag — no spike, no branch)

1. CREATE with the stated falsifier (design-gap vs shadowing-bug vs miscounted-metric).
2. Gates: build 54/54; tests 141/141; `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. Story-shape + 2k-shape TGE_GPU_TRACE runs; aggregated per-phase (§6).
4. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (story 13 tok: 172-173 calls; 2k run: 5216 calls)

| phase | story total | 2k total | share (2k) | reading |
|---|---|---|---|---|
| alloc (descriptors+record+input memcpy) | 5.6 ms | 224.0 ms | ~2% | small; not the lever |
| h2d (weight upload, miss-only) | 412.1 ms | 438.3 ms | ~3% | first-touch sum; p50 0.000 = cache HITS |
| submit (kernel + full fence wait) | 67.5 ms | 3004.7 ms | ~23% | bundled; needs GPU timestamps to split |
| d2h (output readback) | 32.7 ms | **9557.9 ms** | **~72%** | uncached WC reads; THE cost to fix |

## 7. Full reimplementation recipe (triage handoff — NOT built)

For whoever owns the GPU backend: (1) confirm with a direct test that `g_io_out_mem` lands on a write-combined type on the RX 9060 XT (print `propertyFlags`); (2) if so, allocate the output buffer (or a double-buffered readback path) with HOST_CACHED host-visible memory, or copy device-local → cached readback explicitly; (3) re-run the §6 trace and expect d2h to collapse toward memcpy rates, which alone would decide whether the offload can win. Do NOT bother with: re-upload suspicions (disproven), submit-side micro-opts first (smaller share, and bundled with kernel time). Revival trigger for a re-measure: any change to I/O buffer placement, batching of submits, or pipelined (fence-free) dispatch.

## 8. Verdict and why

**TRIAGED, with my own prior headline corrected on the record: weights persist fine; the output readback path is the cost center (~72%), consistent with uncached WC reads.** This closes the loop I opened: validate (sound) → reopen (loses + wrong mechanism hypothesized) → triage (mechanism corrected, fix named, not implemented). The remaining work is owner-side (one memory-type change + timestamped verification), explicitly out of my lane. Next-cycle candidates: the 1517 stage-2 harness remains the only live structural front; the human-gated port shelf now includes the GPU readback fix alongside the rest.

## 9. Exact commands run and their results

- Code re-read (§1 anchors) → cache design found; byte-counter semantics found; per-call path mapped.
- Story + 2k `generate --gpu on` with `TGE_GPU_TRACE=1` → §6 table (no source edits; env flag only).
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
