# SGLang Ascend NPU Communication Lab

Chinese guide: [README_zh-CN.md](README_zh-CN.md)

This directory is a code-reading and experiment companion for adding Chunked
Pipeline Parallelism (CPP) to an SGLang model on Ascend NPU. The experiments use
small tensors first and then the supplied DeepSeek-R1 checkpoint. They are
ordered so that a failure has one narrow communication layer to investigate.

The scripts were syntax-checked without an NPU. Run-time validation must be done
inside the same Ascend/PyTorch/SGLang environment used by `sglang serve`.

## 1. Mental model and learning priority

SGLang has two communication planes:

1. The **control plane** moves request objects, batch metadata, abort/cache
   commands, and final sampled token IDs. It primarily uses CPU process groups,
   broadcasts, and serialized Python objects.
2. The **tensor/data plane** moves activations, partial linear results, KV or
   attention fragments, and MoE tokens. It uses HCCL collectives, P2P tensor
   sends, DeepEP, and optional KV-transfer backends.

For DeepSeek-V4-Flash CPP on NPU, study in this order:

| Priority | Topic | Why it matters |
| --- | --- | --- |
| P0 | Rank topology and process groups | Every collective must use the right TP/PP/DP/CP/EP group, in the same order on every rank. |
| P0 | HCCL semantics, streams, events, buffer lifetime | CPP depends on `isend` being safely deferred while computation and D2H work continue. |
| P0 | PP P2P protocol and scheduler microbatch slots | This is the communication skeleton of CPP. |
| P1 | DP Attention plus DeepEP dispatch/combine | DeepSeek-V4 routes locally held tokens to rank-owned experts at every MoE layer. |
| P1 | Chunked-prefill KV invariants | Later chunks must see earlier chunks' local-stage KV; the KV itself does not cross PP stages. |
| P1 | TP collectives | Attention-TP is 1 in the supplied setup, but dense/shared paths still use the TP group and TP is composed with every PP stage. |
| P2 | CP collectives | Orthogonal to CPP; important when CPP and CP are eventually composed. |
| P3 | PD-disaggregation KV transfer | A separate node-to-node data plane. Learn it only after standalone CPP is correct. |

In the supplied baseline:

```text
tp_size=16, dp_size=16, attn_cp_size=1
attention_tp_size = tp_size / dp_size / attn_cp_size = 1
```

Each rank handles different requests in Attention. MoE then needs DeepEP to
dispatch tokens to the ranks owning the selected experts and to combine the
results back. `--deepep-mode auto` resolves to `normal` for extend/prefill and
`low_latency` for decode.

CPP introduces another dimension. With `tp_size=4, pp_size=4`, 16 NPUs become
four stages, each containing a four-rank TP/DPA/EP group. Hidden-state proxy
tensors travel between stages; DeepEP collectives remain inside each stage.

One easy-to-miss SGLang rule is that enabling DP Attention divides the command
line `chunked_prefill_size` by `dp_size`. The CPP launchers therefore expose
`PER_DP_CHUNK_SIZE` and pass `PER_DP_CHUNK_SIZE * DP_SIZE` to the server. The
value you tune and discuss remains the per-request/per-DP-rank chunk size.

## 2. Experiment 1: HCCL primitives

Run every primitive on all 16 NPUs:

```bash
cd examples/usage/npu_communication_lab
torchrun --standalone --nproc-per-node=16 01_hccl_primitives.py --op all
```

Run one operation with a larger payload:

```bash
torchrun --standalone --nproc-per-node=16 01_hccl_primitives.py \
  --op all_to_all --numel 1048576 --iterations 20
```

Observe the semantic mapping:

- `all_reduce`: TP row-parallel partial sums and many synchronization paths.
- `all_gather`: TP output features, CP K/V, and DP token materialization.
- `reduce_scatter`: the inverse of an all-gather-style replication.
- `all_to_all`: the basic shape of MoE token routing.
- `p2p`: the basic shape of PP activation transfer.

Do not treat the reported number as a production benchmark. The script is for
correctness and relative payload-size experiments.

## 3. Experiment 2: SGLang group topology

This script runs without NPU hardware. First reproduce the supplied baseline:

```bash
python3 02_sglang_topology.py \
  --world-size 16 --tp-size 16 --pp-size 1 \
  --attn-dp-size 16 --attn-cp-size 1 \
  --moe-dp-size 1 --ep-size 16 --rank 7
```

Then inspect a proposed 4-way CPP topology:

```bash
python3 02_sglang_topology.py \
  --world-size 16 --tp-size 4 --pp-size 4 \
  --attn-dp-size 4 --attn-cp-size 1 \
  --moe-dp-size 1 --ep-size 4 --rank 7
```

Before debugging any hang, write down for the failing global rank:

```text
PP group, TP group, Attention-DP index, Attention-TP group, CP group, MoE-EP group
```

A hang where one rank enters a different group or collective order cannot be
fixed in the Attention kernel.

## 4. Experiment 3: Tensor Parallel communication

```bash
torchrun --standalone --nproc-per-node=16 03_tensor_parallel_demo.py
```

The script performs the same two communication shapes that appear repeatedly in
tensor-parallel linear layers:

```text
column-parallel linear: local output columns -> all_gather
row-parallel linear:    local partial sums   -> all_reduce
```

Read next:

- `python/sglang/srt/distributed/parallel_state.py` (`GroupCoordinator`)
- `python/sglang/srt/layers/linear.py`
- `python/sglang/srt/layers/vocab_parallel_embedding.py`

## 5. Experiment 4: MoE dispatch/combine and DeepEP

First learn the data movement without loading a model:

```bash
torchrun --standalone --nproc-per-node=16 04_moe_dispatch_alltoall_demo.py
```

The script implements:

```text
local tokens
  -> router destination
  -> pack by destination
  -> all_to_all dispatch
  -> local expert stand-in
  -> reverse all_to_all combine
  -> restore original token order
```

DeepEP has this semantic role, but uses specialized buffers and kernels and also
carries top-k IDs/weights, token counts, quantization scales, and asynchronous
handles.

Then launch the real DeepSeek-R1 DPA+DeepEP experiment:

```bash
./07_run_dpa_deepep.sh
```

It defaults to 256 chunk tokens per Attention-DP rank. Override that learning
variable with `PER_DP_CHUNK_SIZE`; the launcher multiplies it by DP=16 before
passing the global budget to SGLang.

In a second terminal:

```bash
python3 workload.py --prompt-tokens 2048 --output-tokens 16 \
  --requests 16 --concurrency 16 --profile
```

Compare the modes one at a time:

```bash
DEEPEP_MODE=normal ./07_run_dpa_deepep.sh
DEEPEP_MODE=low_latency ./07_run_dpa_deepep.sh
DEEPEP_MODE=auto ./07_run_dpa_deepep.sh
```

Expected interpretation:

- `normal` is optimized for the many-token prefill/extend shape.
- `low_latency` is optimized for the few-token decode shape.
- `auto` selects between them from `is_extend_in_batch`.

Read next:

- `python/sglang/srt/layers/moe/utils.py` (`DeepEPMode.resolve`)
- `python/sglang/srt/layers/moe/ep_moe/layer.py` (`dispatch -> core -> combine`)
- `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- `python/sglang/srt/hardware_backend/npu/moe/`

## 6. Experiment 5: SGLang PP tensor P2P

Use four NPUs so that each process is one PP stage:

```bash
torchrun --standalone --nproc-per-node=4 05_sglang_pp_p2p_demo.py \
  --chunks 8 --chunk-tokens 128 --hidden 256
```

This experiment initializes SGLang's real PP `GroupCoordinator` and moves a
dictionary shaped like `PPProxyTensors` through the stages. Each stage adds a
visible value to the hidden states. The output demonstrates that:

- stage weights/computation stay in place;
- chunk activations move from stage to stage;
- asynchronous sends need both a later `wait()` and a live reference to the
  payload tensor;
- sender and receiver must agree on tensor keys, shapes, dtypes, and order.

Read next:

- `python/sglang/srt/managers/scheduler_pp_mixin.py`
- `python/sglang/srt/distributed/parallel_state.py` (`send_tensor_dict`, `recv_tensor_dict`)
- `python/sglang/srt/model_executor/forward_batch_info.py` (`PPProxyTensors`)

## 7. Experiment 6: Context Parallel communication

This is not CPP, but it makes the contrast concrete:

```bash
torchrun --standalone --nproc-per-node=4 06_context_parallel_demo.py
```

Each rank owns a contiguous subset of Q/K/V rows. It all-gathers K/V, computes
only its local Q rows with the global causal context, then gathers output rows
for comparison with unsharded Attention.

```text
PP: hidden states cross a stage boundary once per boundary
CP: K/V or partial Attention information communicates inside Attention layers
```

## 8. Experiment 7: fixed-size CPP with the real model

Launch four PP stages with four ranks per stage:

```bash
./08_run_cpp_fixed.sh
```

Send a 6144-token prompt, which should create several 1024-token chunks:

```bash
python3 workload.py --prompt-tokens 6144 --output-tokens 8 --profile
```

Repeat with different chunk sizes:

```bash
PER_DP_CHUNK_SIZE=512  ./08_run_cpp_fixed.sh
PER_DP_CHUNK_SIZE=1024 ./08_run_cpp_fixed.sh
PER_DP_CHUNK_SIZE=2048 ./08_run_cpp_fixed.sh
```

Repeat only after correctness is stable with additional PP output buffering:

```bash
PP_ASYNC_BATCH_DEPTH=2 ./08_run_cpp_fixed.sh
```

Watch or instrument these fields on every PP rank:

```text
pp_rank, mb_id, rid, forward_mode, extend_range,
prefix_lens, extend_lens, seq_lens,
contains_last_prefill_chunk,
PPProxyTensors keys and shapes, out_cache_loc shape
```

Correct invariants are:

1. Every stage sees the same request/chunk identity.
2. For one request, C0 precedes C1 on every stage.
3. `C(k+1).prefix_len == Ck.seq_len`.
4. A stage sends proxy tensors only after its forward event is complete.
5. Each stage retains only the KV belonging to its local layer range.
6. Only the last prompt chunk produces the user-visible first output token.

## 9. Experiment 8: dynamic CPP

Dynamic chunking profiles a cumulative latency curve at startup, so this launch
is deliberately later in the sequence:

```bash
./09_run_cpp_dynamic.sh
```

Then:

```bash
python3 workload.py --prompt-tokens 6144 --output-tokens 8 --profile
```

Compare smoothing factors across separate server runs:

```bash
SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.60 ./09_run_cpp_dynamic.sh
SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.75 ./09_run_cpp_dynamic.sh
SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.85 ./09_run_cpp_dynamic.sh
```

The predictor chooses `x` so that, approximately:

```text
runtime(history + x) - runtime(history) = target chunk latency
```

Later chunks therefore tend to contain fewer tokens. SGLang aligns the result
to a multiple of `max(page_size, 64)`.

## 10. DeepSeek-V4-Flash CPP-specific reading checklist

The current model file already exposes the generic PP contract:

- only the first PP rank owns/uses embeddings;
- only the last PP rank owns final norm and LM head;
- non-first ranks consume `pp_proxy_tensors["hidden_states"]`;
- non-last ranks return flattened mHC hidden states in `PPProxyTensors`;
- weight loading skips layers and boundary weights outside the local PP range.

That means the NPU port should initially be investigated at the boundaries,
rather than by rewriting DSA Attention math:

1. Does every PP stage construct matching DSV4 request/attention metadata?
2. Is flattened mHC `[tokens, hc_mult * hidden]` transferred and reshaped safely?
3. Are DPA/DeepEP collectives scoped inside one TP group/stage?
4. Does a non-first stage avoid dereferencing absent `input_ids` or embeddings?
5. Are DSV4 compressor/indexer/KV locations valid for every middle chunk?
6. Do NPU streams/events protect PP send buffers and DeepEP buffers from early reuse?
7. Is collective order identical on idle, prefill, last-chunk, and decode paths?

Relevant code:

- `python/sglang/srt/models/deepseek_v4.py`
- `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py`
- `python/sglang/srt/hardware_backend/npu/dsv4/`
- `python/sglang/srt/managers/scheduler_pp_mixin.py`
- `python/sglang/srt/managers/schedule_batch.py`

The existing GLM-5.1 NPU registered workload combines `chunked-prefill-size`,
PP8, CP4, DeepEP, and PD disaggregation. It is useful as an advanced reference,
but it is too composite to be the first debugging case:

`test/registered/npu/performance/glm5_1/test_npu_glm5_1_w4a8_1p1d_32p_in64k_out1k_50ms.py`

## 11. Recommended validation ladder for the actual port

Keep the prompt IDs and sampling parameters fixed and compare every step with a
known-good PP1 result:

```text
1. Tiny HCCL primitives
2. SGLang PPProxyTensors P2P demo
3. R1: TP4 x PP4, short prompt, one request
4. R1: TP4 x PP4, long prompt, fixed chunking
5. R1: add concurrent requests
6. R1: add pp_async_batch_depth
7. R1: add dynamic chunking
8. DSV4-Flash: repeat steps 3-7 with eager execution
9. DSV4-Flash: enable NPU graph/multi-stream optimizations
10. Compose CP or PD disaggregation only after standalone CPP is stable
```

For hangs, capture the last entered collective name and group ranks on every
process. For wrong answers, first compare `extend_range`, positions, KV slot
indices, and proxy shapes. For performance gaps, only after correctness, inspect
stream overlap and per-stage duration.
