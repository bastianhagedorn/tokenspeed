# GDN kernels

## FlashInfer PDL adapters

The FlashInfer adapter retains the upstream launch geometry and runtime ABI,
while wrapping device bodies with PDL synchronization. Adapted functions and
in-memory compilation caches live in private namespaces.

Decode and prefill explicitly select FlashInfer's CuTe backend. FlashInfer
0.7.0's automatic backend selection can use Cake GDN for supported shapes,
which bypasses the PDL-wrapped device bodies. Regression tests cover both
power-of-two and other head groupings and reject calls into that alternate path.

FlashInfer 0.7.0 also persists CuTe-DSL kernels to disk. PDL artifacts use a
separate `tokenspeed_pdl_` module namespace so an ordinary kernel cannot satisfy
a PDL cache lookup, or vice versa. Cache invalidation includes all upstream
source files plus the local PDL and GDN adapter sources. Installed FlashInfer
modules and the process-wide cache configuration remain untouched.

GPU regression tests cover persistent cache hits after clearing the in-memory
cache, invalidation after source changes, and both launch orders. Runtime GDN
tests additionally check CUDA Graph dependency edges and exact replay results
for decode, MTP, BF16 state, and prefill.

## Chain verification

ReplaySSM chains use chunked verification on Blackwell for 13–16 tokens and
at least 24 (request, value-head) states. Other shapes and platforms retain
step-by-step verification. `test/runtime/test_gdn_verify_policy.py` checks this
selection and its boundaries without launching device kernels.

## Variable prefill lengths

The Triton L2-normalization fallback uses a runtime token count. The launch
grid and masked block pointers cover the exact rows, while head width and tile
size stay compile-time parameters. Consecutive agentic turns therefore reuse
one compiled kernel across different token counts instead of compiling a new
variant for each length. Tests compare irregular lengths around the usual
prefill graph buckets against an independent PyTorch result.

Compilation guards warm these kernels, then sweep token counts (and PLE
request counts) without allowing new Triton specializations. PLE covers
transitions between single-request, uniform multi-request, and ragged indexing
within one compiled kernel. Those selections use runtime values so startup
warmup also covers batch layouts first seen during serving.

## Accepted-prefix replay

`triton_gdn_replay_commit` commits accepted prefixes for all layers in one
portable Gluon launch. Each CTA owns a layer, request, value head, and V tile.
Keys and gates are prepared in fixed chunks, with K/V staged in shared memory.

State updates stay in FP32 until the final write, and rejected tokens are
masked during loading. Aligned state accesses are vectorized, with fallbacks
for unaligned rows and arbitrary row strides. Batch size, window length, and
payload layer stride remain runtime scalars, so serving shape changes reuse
the compiled kernel.

### Bounded-grid NVIDIA replay

`gluon_nvidia.launch_gdn_replay_commit_persistent` provides an experimental
full-head replay with an explicit CTA budget. A work item owns a complete
128-by-128 value-head state, so V is not split. A persistent grid distributes
layer, request, and head work across the selected CTAs. The CTA budget bounds
occupied SMs; it does not reserve particular SMs or GPCs.

Producer warps stage state and token inputs with asynchronous copies while
compute warps update the accepted prefix in FP32. Full and empty barriers
count every participating thread. Rejected tokens are masked before loading,
odd state rows retain a synchronous-copy fallback, and state conversion happens
only at the final write. Source rows must be independent of other work items'
destinations; in-place updates within one item remain supported.

The opt-in registry entry `gluon_nvidia_gdn_replay_commit` retains a 24-CTA
baseline with two state slots, eight compute warps, and a 168-register limit.
It has lower selection priority than the portable implementation. Select it
explicitly with
`TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_GDN_REPLAY_COMMIT=gluon_nvidia_gdn_replay_commit`.

The [bounded replay benchmarks](../../../../../benchmarks/nvidia/gdn_replay_cta16/README.md)
also save a grouped 16-CTA implementation. For BF16 four-token windows with
three value heads per key head, it shares key normalization and Gram data
across those heads and uses an FP32 projection/triangular solve for accepted
prefixes of three or four tokens. Short prefixes and very small cumulative
decays use the ordinary recurrence. Other supported layouts use the general
replay path. The selected implementation uses two state slots, four producer
warps, and eight compute warps. Batch, layer count, payload stride, and replay
window length stay runtime arguments.

A separate streamed prototype supports up to three state stages. The
[graph overlap and SM placement probe](../../../../../benchmarks/nvidia/gdn_commit_overlap/README.md)
submits the actual captured MTP draft on a second stream and joins it before
other state commits. These are benchmark adapters; stream overlap requires
explicit ordering of the target, draft, and commit.

Run the accepted-state, rejected-input, graph-refresh, and compilation-reuse
checks with:

```bash
PYTHONPATH=tokenspeed-kernel/python:tokenspeed-kernel/test pytest -q \
  tokenspeed-kernel/test/nvidia/ops/test_gdn_replay_pipeline.py
```

The tests retain the repository's dtype tolerances and require untouched state
rows to remain bitwise unchanged. The benchmark README documents standalone
CTA sweeps and the distinction between isolated kernel and end-to-end timings.
