# FORWARD_OP_TABLE — 一个 prefill forward 的算子全景（DSV4.1 / NPU A3 / 4096-token chunk）

> 配套：`KERNEL_CODE_MAP.md`（为什么慢、可改边界）、`profile_bundle/demo_ep_late/`（原始数据）、
> `profile_bundle/callstacks/`（发起位置的栈实证）。
> 本表回答的是另一个问题：**"一次 forward 到底由哪些算子组成、谁发的、吃什么吐什么、干什么用"**。

## 0. 读法与符号

| 标记 | 含义 |
|---|---|
| 【数】 | 形状来自 `indexer_attrs.txt` / `kernel_attrs.txt` 的 Input/Output Shapes（实测） |
| 【推】 | 形状或归属由 config + 代码推出，未经 trace 直接证实 |
| 【未验证】 | 没有证据，只有猜测 |
| 🟢 / 🟡 / 🔴 | 可动性：🟢 Python 层可改｜🟡 要改算子包或 kernel｜🔴 不可动（硬件/语义/已融合） |

符号（来自 `config.json`）：`T=4096`（本 chunk 的 token 数）、`H=5120`、`Nh=64`、`hd=512`、`rope=64`、
`Hc=4`（hc_mult）、`Sink=20`、`q_lora=1280`、`o_groups=8`、`o_lora=1024`、
专家 `E=384`、`topk=6`、`moe_inter=2304`；indexer `32 头 × 128 维`、`topk=512`、候选 `2048 块 × 8`；
压缩页 `64` 行、SWA 页 `128` 行。本 trace 并行：`DP=4 × TP=4`（`--enable-dp-attention --dp-size 4`），
所以 attn 每 die 16 头（64/4），MLA 的 q 就是 `[4096,16,512]`【数】。

---

## 表 A：输入段 + forward 前 metadata（每 forward 一次）

| # | 算子 | 发起位置 | 入参 → 出参 | 作用 | 次/fwd | 实测 | 可动性 |
|---|---|---|---|---|---|---|---|
| A1 | `GatherV2` | `quantization/unquant.py:953-954 embedding`（← `VocabParallelEmbedding`，`MODEL:3954`） | `w[129280,5120]bf16` + `ids[4096]int64` + axis → `h[4096,5120]bf16`【数】 | 词嵌入查表 | 1 | 0.079 ms | 🟢 |
| A2 | `_hc_broadcast_kernel` | `kernels/ops/layernorm/mhc.py:1863 hc_broadcast` ← `MODEL:4631` | `[4096,5120]` → `[4096,4,5120]`（4 份完全相同）【推，语义见 `mhc.py:1855`】 | 造出 4 条 hyper-connection 残差流的**初始状态** | 1 | 0.325 ms | 🟡 triton |
| A3 | `RepeatInterleave` | `layers/engram.py:439 forward` ← `MODEL:4203` | 【未验证】 | **未验证**（疑似 engram 的 n-gram 展开） | 1 | 0.519 ms | — |
| A4 | `SparseFlashMlaMetadata` ×3 | `BACKEND:2074 _dsv41_attention_kernel_metadata` ← `:4527 _compute_kernel_metadata` ← `:4433 init_forward_metadata` | `cu_seqlens_q`、`seqused_k`、`block_table` → dispatch table `int32[1024]` ×3（win / c1 / c2）【推】 | attention 算子的 tiling/dispatch 表，forward 前一次建好、层内复用 | 3 | 2.3 ms | 🟡 |

### A1/A2 的两个细节（值得记住）

- **A1 在本部署里没有做词表切分**。`--enable-dp-attention` ⇒ `MODEL:3957` 的 `enable_tp=not is_dp_attention_enabled()`
  取 False ⇒ `vocab_parallel_embedding.py:259-262` 令 `tp_size=1` ⇒ 表是完整的 `[129280,5120]`。
  代价：**每 die 一份 1.32 GB 词表（16 die ≈ 21 GB 复制）**；好处：这个算子后面**不需要 all-reduce**。
  （若哪天真开了 vocab parallel，形状会变成 `[129280/4,5120]` 并多一次 all-reduce。）
- A1 的耗时是纯带宽：4096 行 × 10 240 B ≈ 40 MiB / 78.6 µs ≈ **533 GB/s**（≈1/3 HBM 峰值）。
  只嵌 4096 个**新增** token，与前缀长度无关 ⇒ 这一项**不随上下文增长**。
- A2 之所以是 A1 的 4 倍（0.325 vs 0.079 ms）：它要**写 4 份**（`mhc.py:1842-1851` 的 triton kernel
  对每行做 `HC` 次 store）。这是 mHC 的第一笔"税"，也是 `Hc=4` 的直接成本。

---

## 表 B：每层主干（×40 层；每层含 2 个 sublayer，多数算子 ×2）

| # | 算子（profiler） | 发起位置（栈实证） | 入参 → 出参 | 作用 | 次/层 | 次/fwd | 实测/fwd | 可动性 |
|---|---|---|---|---|---|---|---|---|
| B1 | `ReduceSum`/`Adds`/`RealDiv`/`SoftmaxV2` | `dsv4_mhc.py:30-35 mix_stats_torch` | `[T,4·5120]fp32` → `pre[T,4]`、`post[T,4]`、`comb[T,4,4]` | **mHC 混合系数**：rms 归一 + sigmoid + 19 轮 Sinkhorn | 2 | 3282 / 3774 / 3200 / 80 | 85 / 91 / 41 / 2.8 ms | 🟢（改语义）/ 🟡（要融合） |
| B2 | `Pows`/`ReduceMean`/`Rsqrt`/`Mul` | `MHC_T:23-24` | `[T,4·5120]` → `[T,1]` → 广播 | B1 的 rms 归一（`x²→mean→+eps→rsqrt→×`） | 2 | 88 / 91 / 88 / 667 | 49 / 32 / 3.5 / 77 ms | 🟢 |
| B3 | `BatchMatMulV2`（fp32 bmm） | `MHC_T:47 post_torch` | `combᵀ[T,4,4]fp32 × residual[T,4,5120]fp32` → `[T,4,5120]` | **残差混合**（mHC 的 post） | 2 | 160 | **≈176 ms** | 🟢/🟡 最大单笔非 indexer |
| B4 | `RmsNorm`(+`DynamicQuant`) | `deepseek_v4.py:622 _fused_rmsnorm_fp8_quant` / `torch_npu.npu_rms_norm` | `[T,5120]` → `[T,5120]`（+per-token scale） | 子层输入归一（可顺带量化） | ~4 | 169 | 9.9 ms | 🔴 已融合 |
| B5 | `MatMulV3`/`QuantBatchMatmulV3` | `quantization/linear_method_npu.py:141 apply` | `q_a:[T,5120]→[T,1280]`；`kv_a:[T,5120]→[T,576]`；`wq_b:[T,1280]→[T,64·512]`【推】 | QKV 低秩投影（w4a8） | 2–3 | 85 / 128 | 64 / 23 ms | 🔴 w4a8 |
| B6 | `InplacePartialRotaryMul` | `dsv4_rope.py:165,177 apply_rotary_mul_inplace` ← `MODEL:2522` | `q/kv[T,h,d]` + `cos/sin[T,64]` → 原地 | RoPE（只转 nope 部分） | 2–3 | 139 | 5.1 ms | 🔴 已融合 |
| B7 | `ScatterUpdate` | KV 写池（`store_kv`） | `kv[T,1,576]` → PA_ND 池 | 把本 chunk 的 KV 写进 paged cache | ~1 | 48 | 13.3 ms | 🔴 |
| B8 | `CompressorV2`（仅 ratio-2 源层 2/8/14） | `BACKEND:1340 _low_ratio_compress`（A3: `_dsv41_a3_compressor`） | `[T,5120]`+`wkv/wgate[5120,512]` → `latent[T,512]`（写 c2 池） | 2:1 压缩 KV（带 gate/配对） | 0.075 | 3 | 1.9 ms | 🟡 |
| B9 | 普通 compressor（ratio-1 源层 20） | `BACKEND:1348`（`ratio==1` 无 gate） | `[T,5120]→latent[T,512]` | 1:1 压缩（就是一次投影） | 0.025 | 1 | 计在 B4/B5 | 🟢 |
| B10 | `LightningIndexerV2`（层 2/8/14/20） | `BACKEND:1611`（plain）/ `:1438`（candidate） | `q[T,32,128]`+`k[c2池 2033,64,1,128]`+`w[T,32]`+`bt[1,960]` → `idx[T,512]`（+`cand[T,1,2048]`）【数】 | **一级**：全量 top-512；层 20 额外发 2048 候选块 | 0.1 | 4 | 166 ms（3×28+1×81） | 🟡 见 E2/E3 |
| B11 | `SparseLightningIndexer`（层 24/28/32/36） | `BACKEND:1755 _dsv41_index_qsli_topk` | `q[T,32,128]`+`k[c1池 4065,64,1,128]`+`w`+`cand[T,1,2048]`+`bt[1,1920]` → `idx[T,512]`【数】 | **二级**：候选块内 rescore（leak 语义仍需扫全 S2） | 0.1 | 4 | **4000 ms** | 🟡/🔴 |
| B12 | `SparseFlashMla` | `BACKEND:2214 _forward_low_ratio_attention` | `q[T,16,512]`+`ori_kv[642,128,1,512]`+(`cmp_kv`+`idx[T,1,512]`)+`bt` → `out[T,16,512]`【数】 | 稀疏 MLA（window 128 + 512 个压缩位置） | 1 | 40 | 148.6 ms | 🔴 |
| B13 | `BatchMatMulV2`（bf16 einsum） | `MODEL:619 _apply_wo_a_bf16_matmul` | `o[T,8,128] × wo_a[8,512,128]` → `[T,8,512]`【推】 | o_proj 的 grouped 低秩上投影 | 1 | 80 | ≈24 ms | 🟢 |
| B14 | `MoeGatingTopK` | `srt/layers/moe/topk.py`（router） | `logits[T,384]` → `ids[T,6]`、`w[T,6]` | 选 6/384 个专家 | 1 | 40 | 1.5 ms | 🔴 |
| B15 | `DispatchLayout` → `NotifyDispatch` → `CamMoeDispatchNormal` | `DISPATCH:648` / `:668` | `x[T,5120]`、`ids[T,6]` → `recv_x[?]`、`handle` | MoE all-to-all 前置布局 + 发送 | 1 | 40 各 | 1.4 / 24.5 / 23.9 ms | 🟡 |
| B16 | `GroupedMatmul` ×2 + `ClippedSwiglu`(+quant) | `layers/moe/matmul.py:41` ← `moe_runner/ascend.py:278/310`；`npu/moe/activation.py:66-72` | `gate_up[T,5120]→[T,2·2304]`；`down[T,2304]→[T,5120]` | 384 专家的 w4a8 GEMM + SwiGLU | 2 | 80 / 80 | 30 / 1.9 ms | 🔴 |
| B17 | `CamMoeCombineNormal` | `DISPATCH:723 _combine_core`（**normal 模式**） | `recv[1333,5120]…` → `[1333,5120]`【数】 | MoE 回收 + 按权重求和 | 1 | 40 | 70.9 ms | 🟡 |

（B15/B16/B17 的 `?`/1333 是**每个 rank 实际收到的 token 数**，随层抖动 ⇒ 时长波动，见 `KERNEL_CODE_MAP` Q6。）

---

## 表 C：只有特定层/位置才有的算子（"为什么是 4+4 / 3 / 2"的答案）

| 算子 | 只在哪些层 | 依据 | 次/fwd |
|---|---|---|---|
| `CompressorV2` | 2, 8, 14（ratio 2 且 ∈ `kv_source_layer_ids`） | `IDX:224` + `BACKEND:1340` | 3 |
| 普通 compressor（ratio 1） | 20 | `BACKEND:1348`（ratio==1 无 gate） | 1 |
| `LightningIndexerV2` | 2, 8, 14（plain）+ 20（candidate） | `BACKEND:1435/1445` | 4 |
| `SparseLightningIndexer` | 24, 28, 32, 36 | `IDX:226` `uses_candidates` | 4 |
| `LeftShift`（AI_CPU，9.5 ms） | 1, 14（engram 层） | `engram.py:122 _e8m0_to_float` | 2 |
| `RepeatInterleave` | 层循环入口（1 次） | `engram.py:439` ← `MODEL:4203` | 1 |
| `_hc_broadcast_kernel` | forward 入口 | `MODEL:4631` | 1 |
| `SparseFlashMlaMetadata` | forward 前 | `BACKEND:2074` | 3 |

---

## 表 D：尾部（每 forward 一次，**未采栈** ⇒ 全部【未验证】）

`RmsNorm` → `hc_contract`（`mhc.py:1869-1870`，4 条残差流取平均合回 1 条）→ `lm_head`（`[5120]→[129280]`）
→ sampler（`TopK`/`TopP`/`Multinomial`）。目前只核到 `MODEL:4795 self.norm(...)`，其余待补。

---

## 待补清单

1. 表 D 的尾部算子（`lm_head` / sampler）——补一条 `--ops-re 'lm_head|Linear|TopK|Softmax'` 的栈即可。
2. `RepeatInterleave`（A3）到底在干什么——它每 forward 一次、0.5 ms，目前只有"engram"这个线索。
3. 表 B 里"次/层"带 `~` 的（B4/B6/B7）需要按层枚举才能精确；现在是从 `op_statistic` 总数除出来的。
