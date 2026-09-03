# SGLang Ascend NPU 通信实验手册

这组实验服务于一个具体目标：在 Ascend NPU 上理解、验证并最终适配
DeepSeek-V4-Flash 的 Chunked Pipeline Parallelism（CPP）。详细代码索引和
英文说明见 [README.md](README.md)。

这些脚本已经在无 NPU 的开发机上通过 Python 和 Shell 语法检查；HCCL、
DeepEP 和真实模型的运行结果仍需在安装了 `torch_npu` 的 16 卡环境验证。

## 一、先建立通信框架的整体认识

SGLang 的通信不是一个单独的模块，可以按下面五层理解：

1. **Rank 与通信组**：把全局 rank 组织为 TP、PP、Attention-DP、CP、
   MoE-EP 等不同的 group。所有 rank 必须以相同顺序进入相同 collective。
2. **基础通信原语**：HCCL 提供 broadcast、all-reduce、all-gather、
   reduce-scatter、all-to-all 和 send/recv。
3. **模型并行语义**：TP 传部分线性层结果，CP 传 Attention 上下文，EP/
   DeepEP 路由 MoE token，PP 传 stage 边界的 hidden states。
4. **调度与控制面**：scheduler 传 request、batch metadata、缓存控制和采样
   结果；CPP 在这里把 chunk 组织为连续流动的 microbatch。
5. **专用数据面**：DeepEP 负责 MoE dispatch/combine；PD disaggregation 的
   KV transfer 则是另一条、通常跨节点的数据面。

对于你的 DSV4 CPP 工作，建议优先级为：

```text
P0  group/rank 拓扑
P0  HCCL 原语、stream/event、异步 buffer 生命周期
P0  PP 的 request/proxy/output 三类通信和 microbatch 槽位
P1  Attention-DP + DeepEP dispatch/combine
P1  chunk 的 position、prefix_len、KV slot 不变量
P1  TP collective（dense/shared 路径仍会用，且需与每个 PP stage 组合）
P2  CP 的具体 collective
P3  PD disaggregation/KV transfer
```

你提供的基线配置中：

```text
TP=16, Attention-DP=16, Attention-CP=1
Attention-TP = 16 / 16 / 1 = 1
```

因此每张卡的 Attention 处理不同请求，而不是 16 张卡共同算同一请求的
Attention。进入 MoE 层后，token 再通过 DeepEP 被发往持有所选 expert 的
rank，计算完成后再返回原 token 所在 rank。

## 二、实验顺序

进入目录：

```bash
cd examples/usage/npu_communication_lab
```

### 实验 1：HCCL 基础原语

```bash
torchrun --standalone --nproc-per-node=16 01_hccl_primitives.py --op all
```

目标是分别确认 collective 和 P2P 的输入输出语义，不是得到正式性能数据。
每个 op 最终都应由 rank 0 打印 `PASS`。还可以单独改变 payload：

```bash
torchrun --standalone --nproc-per-node=16 01_hccl_primitives.py \
  --op all_to_all --numel 1048576 --iterations 20
```

### 实验 2：SGLang 通信组拓扑

此脚本不需要 NPU。先打印你当前基线中 rank 7 所属的 group：

```bash
python3 02_sglang_topology.py \
  --world-size 16 --tp-size 16 --pp-size 1 \
  --attn-dp-size 16 --attn-cp-size 1 \
  --moe-dp-size 1 --ep-size 16 --rank 7
```

再观察 TP4 × PP4：

```bash
python3 02_sglang_topology.py \
  --world-size 16 --tp-size 4 --pp-size 4 \
  --attn-dp-size 4 --attn-cp-size 1 \
  --moe-dp-size 1 --ep-size 4 --rank 7
```

第二个配置中 rank 7 位于：

```text
TP/EP group: [4, 5, 6, 7]       # PP stage 1 内部
PP group:    [3, 7, 11, 15]     # 同一个 TP lane 穿过四个 stage
Attention-TP/CP group: [7]      # Attention 实际没有跨卡 collective
```

### 实验 3：Tensor Parallel

```bash
torchrun --standalone --nproc-per-node=16 03_tensor_parallel_demo.py
```

它用一个小矩阵展示两种最常见模式：

```text
Column Parallel: 本地输出列 -> all-gather
Row Parallel:    本地部分和 -> all-reduce
```

两条路径都会与未切分矩阵乘法比较，误差超限会直接失败。

### 实验 4：MoE All-to-All 与 DeepEP

先运行不依赖具体模型的 top-1 token 路由：

```bash
torchrun --standalone --nproc-per-node=16 04_moe_dispatch_alltoall_demo.py
```

脚本会执行 pack、all-to-all dispatch、本地 expert 占位计算、反向
all-to-all combine、恢复原 token 顺序，并检查结果。DeepEP 的语义相同，
但还会管理 top-k、scale、容量、量化格式、专用 buffer 和异步 handle。

再启动真实 DeepSeek-R1 DPA+DeepEP：

```bash
./07_run_dpa_deepep.sh
```

默认每个 Attention-DP rank 的 chunk 上限是 256 token，可用
`PER_DP_CHUNK_SIZE` 修改；启动脚本会自动乘以 DP=16 后传给 server。

另一个终端发送 16 个并发请求：

```bash
python3 workload.py --prompt-tokens 2048 --output-tokens 16 \
  --requests 16 --concurrency 16 --profile
```

分开重启并比较：

```bash
DEEPEP_MODE=normal ./07_run_dpa_deepep.sh
DEEPEP_MODE=low_latency ./07_run_dpa_deepep.sh
DEEPEP_MODE=auto ./07_run_dpa_deepep.sh
```

`auto` 在 extend/prefill 使用 normal，在 decode 使用 low_latency。

### 实验 5：真实 SGLang PP P2P

```bash
torchrun --standalone --nproc-per-node=4 05_sglang_pp_p2p_demo.py \
  --chunks 8 --chunk-tokens 128 --hidden 256
```

它初始化 SGLang 自己的 PP `GroupCoordinator`，用 `PPProxyTensors` 的形状
传递 hidden states 和 residual。`send_tensor_dict` 内部同时传输 CPU 元数据
与 NPU tensor，因此也能观察控制信息和数据面的配合。最后一个 stage 应打印：

```text
All chunks crossed every PP stage: PASS
```

实验特意保留异步 send 的 work 和 payload，直到 `wait()` 完成。CPP 适配中
若发送 tensor 已被下一轮复用，通常会表现为偶发错值，而不是稳定报错。

### 实验 6：Context Parallel

```bash
torchrun --standalone --nproc-per-node=4 06_context_parallel_demo.py
```

每个 rank 持有连续的一段 Q/K/V，all-gather 完整 K/V 后只计算本地 Q 行，
最后与未切分的 causal Attention 比较。这个实现有意保持朴素，用来强调：

```text
PP/CPP：hidden states 在 layer stage 之间移动
CP：    K/V 或 Attention 中间结果在 Attention 内部移动
```

### 实验 7：固定 chunk 的真实 CPP

```bash
./08_run_cpp_fixed.sh
```

另一个终端发送一条 6144-token prompt：

```bash
python3 workload.py --prompt-tokens 6144 --output-tokens 8 --profile
```

默认是 TP4 × PP4 × Attention-DP4，且每个 DP rank 的 chunk 为 1024 token。
注意：SGLang 开启 DP Attention 后会把命令行 `chunked_prefill_size` 除以
`dp_size`，所以启动脚本内部实际向 server 传 `1024 * 4`。

分开重启比较不同粒度：

```bash
PER_DP_CHUNK_SIZE=512  ./08_run_cpp_fixed.sh
PER_DP_CHUNK_SIZE=1024 ./08_run_cpp_fixed.sh
PER_DP_CHUNK_SIZE=2048 ./08_run_cpp_fixed.sh
```

正确性稳定后再增加在途 batch 深度：

```bash
PP_ASYNC_BATCH_DEPTH=2 ./08_run_cpp_fixed.sh
```

### 实验 8：动态 CPP

```bash
./09_run_cpp_dynamic.sh
```

动态模式启动时会采样一条累计 prefill 延迟曲线，拟合
`f(L)=aL^2+bL+c`，再为已有历史长度 `L` 选择下一段 `x`，使：

```text
f(L + x) - f(L) ≈ 目标 chunk 时间
```

所以请求越往后，chunk 通常越小。可分别重启比较平滑系数：

```bash
SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.60 ./09_run_cpp_dynamic.sh
SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.75 ./09_run_cpp_dynamic.sh
SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.85 ./09_run_cpp_dynamic.sh
```

## 三、CPP 调试时要记录的字段

在每个 PP rank 上记录：

```text
pp_rank, mb_id, rid, forward_mode,
extend_range, prefix_lens, extend_lens, seq_lens,
contains_last_prefill_chunk,
PPProxyTensors 的 key/shape/dtype,
out_cache_loc 与 out_cache_loc_dsv4
```

逐项验证：

1. 所有 stage 看到相同的 request/chunk 身份。
2. 同一请求在每个 stage 上都是 C0、C1、C2 的顺序。
3. `C(k+1).prefix_len == Ck.seq_len`。
4. forward event 完成后才能发送本 stage 的 proxy tensor。
5. KV cache 不跨 PP stage；每个 stage 只保留本地 layers 的 KV。
6. 只有 prompt 的最后一个 chunk 才产生用户可见的第一个输出 token。
7. 空 batch、普通 chunk、最后 chunk、decode 分支中的 collective 顺序一致。

## 四、面向 DeepSeek-V4-Flash 的代码入口

当前 `deepseek_v4.py` 已经接入通用 PP 边界：第一 stage 负责 embedding，最后
stage 负责 norm/lm-head，中间 stage 收发展平后的 mHC hidden states。因此 NPU
适配应优先检查边界与状态，而不是先改 DSA Attention 数学：

```text
python/sglang/srt/managers/scheduler_pp_mixin.py
python/sglang/srt/distributed/parallel_state.py
python/sglang/srt/model_executor/forward_batch_info.py
python/sglang/srt/models/deepseek_v4.py
python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py
python/sglang/srt/hardware_backend/npu/dsv4/
python/sglang/srt/layers/moe/token_dispatcher/deepep.py
python/sglang/srt/hardware_backend/npu/moe/
```

建议最终按下面的阶梯验证：

```text
HCCL 小 tensor
-> SGLang PPProxyTensors P2P
-> R1 TP4×PP4 短 prompt 单请求
-> R1 固定 chunk 长 prompt
-> 并发请求
-> pp_async_batch_depth
-> dynamic chunking
-> DSV4-Flash eager 模式重复上述步骤
-> 最后打开 NPU graph/multi-stream，并再组合 CP 或 PD
```

PP 实验没有保留原命令中的 NEXTN 参数，因为当前 SGLang 明确禁止 PP 与
speculative decoding 组合。先把 CPP 单独跑正确，再讨论是否需要扩展该约束。
