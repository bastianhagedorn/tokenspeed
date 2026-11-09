# Shared PLE host tables

PLE can keep one complete host table per node and pipeline stage, shared by
all of that stage's local workers. Each worker looks up only its own n-gram
IDs: neither DP ID all-gather nor TP/DP result reduction is needed. TP peers
can therefore read the same host rows independently. This trades collective
traffic for direct host-memory traffic; it is not a guaranteed throughput win.

## Explicit placement

`ple_offload_embedding` retains its existing boolean behavior: `false` keeps
device table shards, `true` keeps private pinned-host shards. The shared modes
are explicit string values:

| Value | Shared allocation |
| --- | --- |
| `"shm"` | `/dev/shm` mapping, registered separately in each worker |
| `"cuda_vmm"` | Host-NUMA CUDA VMM with FABRIC handles; fail if unavailable |
| `"auto"` | Try host-NUMA VMM, fall back consistently to `/dev/shm` |

For example, merge into the deployment's existing text HF overrides:

```json
{"ple_offload_embedding":"auto","ple_embed_dtype":"float8_e4m3fn"}
```

The complete table must fit both the node's available host memory (bounded
by the process's cgroup and ancestor limits) and, for the shm fallback, free
`/dev/shm` capacity. Container deployments may need a larger `--shm-size` or
a shared IPC namespace. There is no silent disk or NVMe fallback. Different
nodes own independent complete copies; this is not cross-node shared RAM.

## Large pages are separate from sharing

The NVIDIA allocation uses host-NUMA pinned physical allocations, at least
2 MiB allocation granularity, mapped into each worker's CPU/GPU address space.
Creation and FABRIC export are probed rather than inferred from GPU generation.
Allocations are chunked, leaving a 5% NUMA-node reserve; any rank's allocation,
import or mapping failure makes every participant abandon that attempt.
`auto` then falls back, while `cuda_vmm` reports an error.

The shm backend requests transparent huge pages before registration, but
this is only an OS hint. Its log reports `gpu_page_bytes=0` (unknown), not
a promise of 2 MiB GPU mappings. Before registration, workers exchange GPU host
NUMA nodes. The shm mapping prefers that node when all local GPUs share one;
otherwise it interleaves pages across their host nodes, rather than depending
on the CPU that first faults a page. All local workers apply the same policy.
If GPU topology is unavailable, `mbind(MPOL_LOCAL)` is attempted instead.
Policy failures are logged and the inherited policy remains. This does not
change process CPU affinity, migrate existing pages, or make shared pages local
to every GPU. VMM retains its explicit writer-GPU host-node placement.
Memory prechecks reduce risk,
but cannot reserve capacity against other processes allocating concurrently.

## Loading, lookup, and ownership

Only the lowest local stage rank writes checkpoint rows. Other workers still
consume checkpoint metadata but never overwrite the shared table. Offline FP8
uses the checkpoint-wide scalar independently on each worker. Online FP8 is
quantized by the writer; its per-row scales are published in another shared
host allocation and accessed through zero-copy device aliases, not replicated
full-table GPU scale buffers.

All stage workers call `finish_loading()` once at the end of the Qwen weight
loader. It validates scale metadata and publishes completed weights/scales.
No startup collective, registration or allocation happens inside the existing
`start`/`finish` lookup lifecycle or during CUDA graph replay. The host gather
continues on its side stream with its existing completion event and PDL behavior.
Shared tables are immutable after publication: live weight reload and dummy
weight initialization are not supported by these explicit shared modes.

Each node's writer creates a unique, sparse file. Every peer opens its own fd;
the writer then unlinks the file before any mmap registration can populate it.
Closing or killing a worker cannot orphan a fully populated named shm table.
The CPU tensor and device alias retain the backing allocation until their last
storage view dies. CUDA registration and VMM teardown occur only after device
work has completed. Failed setup is cleaned up on every surviving participant.

## Gather launch geometry

Queries below 256 flattened n-gram IDs retain one row and one warp per CTA;
other queries use four rows and four warps, except for large complete tables.
Sharded queries retain the direct four-row path: sorting did not consistently
improve blocks from actual agentic requests, while persistent launch submission
costs outweighed their GPU gains in eager prefill.
Complete-table queries at 131,072 IDs sort IDs on the GPU and scatter results
back to their original order, using 32-row tiles and eight warps with the grid
capped at the SM count. The caller supplies the table partition count
explicitly, so pointer representation does not determine table ownership.

Launch geometry is checked against both uniformly sampled IDs and checkpoint
hashes of tokenized agentic inputs. Repeated natural-language n-grams can make
a geometry that wins on uniformly random IDs slower in a serving trace.

Sorting retains 64-bit IDs and adds temporary ID/index buffers and an eager
CPU submission cost. Kernel measurements must include sorting and scattering;
end-to-end measurements also include that CPU cost and the existing overlap.
Token counts and the grid stride remain runtime arguments. Only the bounded
launch configurations and model dimensions specialize the kernels. Tail rows
and out-of-shard table and scale loads remain masked.

The conv kernel writes rollback windows as channel-by-state tiles. This
coalesces the state dimension while preserving the existing fused launch,
epilogue rounding, carried/final-state handling and PDL synchronization.

The allocation and registration contracts follow the CUDA Driver API's
[virtual memory management](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html)
and [host memory registration](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html)
interfaces. GPU support and observed page sizes still require validation on
the deployment's driver and hardware; CPU sharing tests do not establish them.
