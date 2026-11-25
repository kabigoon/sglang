# KERNEL_CODE_MAP — DeepSeek-V4.1-Flash(w4a8) / NPU A3×16 / prefill 算子 → sglang 代码路径

> 目的：把 `profile_bundle/demo_ep_late` 里的算子名逐一对上代码坐标，为"该改哪里"提供可复核的锚点。
> 数据：`_tmp_preflight/profile_bundle/demo_ep_late/`（2 个 forward，每个 4096 token，T≈120K）。
> 代码：`/Users/an/Developer/sglang` @ `pr-50205-dsv41-a3-deepep`。
> 算子实现：`/Users/an/Developer/ops-transformer`（CANN ops-transformer 9.2.0 快照，**不是** 机器上那个 wheel）、
> `/Users/an/Developer/sgl-kernel-npu`（deep_ep / CamMoe 源）。
> 机器上没有 NPU 环境，所以 **wheel 的二进制内容一律标「未验证」**；凡是只依赖仓库源码 + 本次 trace 的结论，都可复核。

## 0. 缩写与证据等级

| 缩写 | 全路径 |
|---|---|
| `BACKEND` | `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py` |
| `IDX` | `python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py` |
| `MODEL` | `python/sglang/srt/models/deepseek_v4.py` |
| `MHC` | `python/sglang/srt/models/deepseek_v4_mhc.py` |
| `MHC_T` | `python/sglang/srt/hardware_backend/npu/dsv4/dsv4_mhc.py` |
| `POOL` | `python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py` |
| `DISPATCH` | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` |
| `OP_SLI` | `ops-transformer/experimental/attention/sparse_lightning_indexer`（= profiler 里的 `SparseLightningIndexer`） |
| `OP_LIV2` | `ops-transformer/experimental/attention/lightning_indexer_v2`（= profiler 里的 `LightningIndexerV2`） |

证据等级：**【码】**=仓库/算子源码直接读到；**【数】**=本次 bundle 的数据；**【推】**=由【码】+【数】推出（给出算式）；**【未验证】**=本机无法核对。

**profiler 名 ↔ 算子源码的桥**：`kernel_details.csv` 的 `OP Type` 就是 aclnn 算子名。
`SparseLightningIndexer` = `OP_SLI/op_host/sparse_lightning_indexer_def.cpp:19,128`；`LightningIndexerV2` = `OP_LIV2/op_host/lightning_indexer_v2_def.cpp:18,123`。两者都只注册 `ascend910b` / `ascend910_93`（arch22，即 A2/A3）——**没有 arch35 配置**，与 `is_npu_arch35()==False` 的 A3 分支严格对应（见 Q1）。

---

## 1. 映射总表（每 forward = 每个 4096-token prefill chunk）

次数 = `op_statistic.csv` 的 `Count / 2`；耗时 = `Total / 2`。

| 算子（profiler OP Type） | Python 发起位置（文件:行） | 次数/forward | 触发条件 | 实测/forward | 备注 |
|---|---|---|---|---|---|
| **SparseLightningIndexer** | `BACKEND:1755`（`_dsv41_index_qsli_topk` 的 `if not is_npu_arch35():` 分支） | **4** | `indexer.uses_candidates == True`（`IDX:226`：`0 <= candidate_source_layer_id < layer_id`）→ layer 24/28/32/36；且 arch≠35 | **4.00 s**（4×~1000 ms） | 第 2 级 topk（consumer）。`quant_sparse_lightning_indexer_dsl`（`BACKEND:1780`）是 **A5 路径，本次运行没走**，见 Q1 |
| **LightningIndexerV2** | `BACKEND:1599`（`candidate=True`）/ `BACKEND:1611`（plain） | **4** | 在 `index_source_layer_ids` 且 `not uses_candidates` → layer 2/8/14/20 | **165.6 ms** = 3×28.02 + 1×**81.49** | layer 20 那次是 candidate 发布者（多个 candidate 输出），仍只要 81.5 ms |
| **SparseFlashMla** | `BACKEND:2214`（A3）；A5 在 `BACKEND:2287` | **40** | 每层 1 次；`compress_ratio==0` 层只带 window，ratio 1/2 层额外带 `cmp_kv` + `c{ratio}_topk_indices` | **148.6 ms**（38×3.72 + 2×0.21） | 真注意力，与 indexer 无关；不是瓶颈 |
| SparseFlashMlaMetadata | `BACKEND:2074`（A3，`_dsv41_attention_kernel_metadata`） | 3 | 每个 ratio（0/1/2）各一份 dispatch table | 2.3 ms（AI_CPU，avg 767 µs） | 3 = win + c1 + c2 |
| **CamMoeCombineNormal** | `DISPATCH:956` → `deep_ep.Buffer.low_latency_combine`（`sgl-kernel-npu/python/deep_ep/deep_ep/buffer.py:716`）→ `strategies/low_latency_strategy.py:150` → C++ runtime（deep_ep ops2）→ `sgl-kernel-npu/csrc/deepep/ops2/op_host/cam_moe_combine_normal.cpp:4,85`（`OP_ADD(CamMoeCombineNormal)`） | **40** | EP=deepep + low-latency + 每层 MoE | **70.9 ms**（avg 1.77 ms，min 0.083 / max 5.93） | 是 DeepEP 的 combine（P2P），不是 HCCL；见 Q6 |
| CamMoeDispatchNormal / NotifyDispatch / DispatchLayout | 同一 dispatch 链（`DISPATCH` 的 low-latency dispatch） | 40 / 40 / 40 | 同上 | 23.9 / 24.5 / 1.4 ms | ops2 的三个配套 kernel |
| **Cast**（`aclnn…_CastAiCore_Cast`） | 主源：`MHC_T:22,23,47-49`（fp32 往返）、`BACKEND:1512`（NZ format cast）、MoE 量化 | **1087** | — | 149.1 ms | 单笔最大 498 µs |
| **Add**（含 `Adds`） | `MHC_T:33-35`（sinkhorn eps）、`MHC_T:45-48`（残差）、bias | **3774** | sinkhorn_iters=20 | 91.0 ms | ~3120 次来自 sinkhorn |
| **ReduceSum** | `MHC_T:33-35`（`comb.sum(dim=-1/-2)`） | **3282** | 同上 | 84.9 ms | 39 次/调用 × 2 调用/层 × 40 层 = 3120（95%） |
| **RealDiv** | `MHC_T:33-35`（`comb / (sum+eps)`） | **3200** | 同上 | 41.0 ms | 与 ReduceSum 一一配对 |
| **Mul** | `MHC_T:24`（`* inv_rms`）、`MHC_T:48`（`post_mix*x`）、RoPE（`Dsv4NpuRoPE.apply_rotary_mul_inplace`） | **667** | — | 77.4 ms | 单笔最大 980 µs（fp32 残差） |
| ReduceMean / Pows / Rsqrt | `MHC_T:23`（`flat.square().mean(-1)` + `rsqrt`） | 91 / 88 / 88 | mHC 的 rms 归一 | 31.9 / 49.1 / 3.5 ms | 与 Mul `inv_rms` 同一行 |
| RmsNorm | `torch_npu.npu_rms_norm`（`BACKEND:1378` compressor 后）等 | 169 | — | 9.9 ms | 融合版，已经不贵 |
| BatchMatMulV2 / MatMulV3 / GroupedMatmul / QuantBatchMatmulV3 | MLA 前后投影 + MoE | 120 / 85 / 80 / 128 | — | 189 / 64 / 30 / 23 ms | 合计 ~306 ms，占 5.6% |

**一句话**：4.00 s（SLI）+ 0.17 s（LIV2）+ 0.15 s（MLA）= 4.32 s，占每 chunk 5.43 s 算子时间的 **79.7%**；其余（elementwise 0.52 s、MoE 通信/combine 0.12 s、matmul 0.31 s）合计 ~1.0 s，都不是主战场。

---

## 2. Q1–Q6 逐条作答

### Q1. `SparseLightningIndexer` 是哪个 Python 函数发起的？

**答案（A3 实际路径）**：`BACKEND:1755`，函数 `_dsv41_index_qsli_topk`（`BACKEND:1729`）。

```
MODEL:2626  attn_backend.forward_low_ratio_sources(...)          # 每个 ratio∈{1,2} 层、attention 之前
 └ BACKEND:1305  forward_low_ratio_sources
    └ BACKEND:1335  self._low_ratio_index_topk(layer, ...)       # run_indexer and layer.indexer is not None
       └ BACKEND:1414  _low_ratio_index_topk
          ├ BACKEND:1435  if indexer.is_candidate_source:  → _dsv41_index_full_topk(candidate=True) → LIV2
          ├ BACKEND:1437  elif indexer.uses_candidates:    → _dsv41_index_qsli_topk          ← 这里
          └ BACKEND:1440  else:                            → _dsv41_index_full_topk          → LIV2
             └ BACKEND:1729  _dsv41_index_qsli_topk
                └ BACKEND:1755  _dsv41_a3_ops().sparse_indexer(...)   # ← SparseLightningIndexer
```

`BACKEND:1755` 原文：

```python
        if not is_npu_arch35():
            indices, _ = _dsv41_a3_ops().sparse_indexer(
                q_packed,
                self.token_to_kv_pool.get_index_k(layer.layer_id),
                w,
                self._dsv4_index_topk,
                candidate_indices,
                candidate_block_size=indexer.candidate_block_size,
                **self._dsv41_a3_index_kwargs(forward_batch, ratio),
            )
```

**关于任务提示里的 1780/1934（重要修正）**：`BACKEND:1780`（`quant_sparse_lightning_indexer_dsl`）与 `BACKEND:1934`
（`quant_sparse_lightning_indexer_metadata_dsl`）都在 `if not is_npu_arch35()` **之后**，是 **A5/arch35 路径**。
本次运行是 **A3 = arch22**，`is_npu_arch35()` 返回 False（`python/sglang/srt/hardware_backend/npu/utils.py:45`
→ `_is_npu_arch(3510)`；A3 是 `is_npu_arch22()` 的 2201）。三条独立证据：

1. 算子名：trace 里是 `SparseLightningIndexer` / `LightningIndexerV2`，且这两个算子源码只注册
   `ascend910b`+`ascend910_93`（`OP_SLI/op_host/sparse_lightning_indexer_def.cpp:124-128`、
   `OP_LIV2/op_host/lightning_indexer_v2_def.cpp:119-123`）；arch35 那对 dsl 算子名为
   `QuantSparseLightningIndexer` / `QuantLightningIndexer`【码】。
2. 数据 dtype：A5 路径的 Q 是 MXFP4 打包（`BACKEND:1570` `npu_dynamic_mx_quant(...float4_e2m1fn_x2)`，
   末维 = `index_head_dim//2` = 64），而 trace 里 q 是 `4096,32,128`、k 末维 128（BF16）【数】。
3. A3 分支的 `_dsv41_index_q` 在非 arch35 时直接 `return q, None, w`（`BACKEND:1568-1569`），
   正好产生 `q[4096,32,128]` / `w[4096,32]`【码】。

所以：**`quant_sparse_lightning_indexer_dsl` 这段代码很重要（是 A5 的同一语义实现），但不是本次 984 ms 的发起者**。

### Q2. 一个 forward 里为什么恰好 4 个 SLI + 4 个 LIV2？

三个 config 字段联合决定，逐层三元组：

- `index_source_layer_ids = [2,8,14,20,24,28,32,36]` → **只有这 8 层**挂 `self.indexer`（`MODEL:1526-1532`）。
  `forward_low_ratio_sources` 里 `if run_indexer and layer.indexer is not None`（`BACKEND:1335`）⇒ 8 层各跑一次 indexer。
- `candidate_source_layer_id = 20` → 只有 layer 20 是 **source**（`IDX:225` `self.is_candidate_source = layer_id == config.candidate_source_layer_id`）。
- `uses_candidates`（`IDX:226` `self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_id`）
  → layer > 20 且带 indexer 的层 = **24/28/32/36** 这 4 层走 QSLI（SLI）。

于是 8 层 = **4 + 4**：

| layer | compress_ratio（config） | indexer | 角色 | 算子 |
|---|---|---|---|---|
| 2, 8, 14 | 2 | 有（且 `owns_k`） | 普通全量 topk | **LightningIndexerV2** ×3 |
| 20 | 1 | 有（`owns_k`，`is_candidate_source`） | 一级：全量 topk + 发候选块 | **LightningIndexerV2** ×1 |
| 24, 28, 32, 36 | 1 | 有（`uses_candidates`，**不** `owns_k`） | 二级：候选块内 rescore | **SparseLightningIndexer** ×4 |

判定条件落点：
- 谁有 indexer：`MODEL:1526`（`if self.layer_id in config.index_source_layer_ids`）。
- 谁发候选：`BACKEND:1435-1441`（`if indexer.is_candidate_source: ... fm.dsv41_candidate_blocks = blocks`）。
- 谁消费候选：`BACKEND:1437` + `IDX:226`。
- 谁 `owns_k`（写 index-K cache）：`IDX:224` `self.owns_k = layer_id in config.kv_source_layer_ids`，
  写 K 在 `BACKEND:1450-1533`（`_dsv41_index_k_store`）。

**为什么 ratio 也重要**：`compress_ratios = [0,0,2×18,1×20,0,0,0]`（config，index 0..42）→
layer 0/1 是 ratio 0（只有 SWA window，不跑 compressor/indexer），layer 2..19 是 ratio 2，layer 20..39 是 ratio 1。
候选源（20）和 4 个消费者（24..36）全在 ratio-1 段，所以两级的 K 都是 **c1 池**（同一分辨率），
这也是 Q3 里那条"两者都扫 122 880 行"的前提。

### Q3. 那"4065 个候选块"到底是什么？

**先纠一个口径错误（这条会改变结论）**：`4065` **不是候选块数**，是 **ratio-1 index-K pool 的 page 数**（k 张量的 dim0）。

- 候选块数 = `candidate_topk_blocks = 2048`/行，就是 SLI 第 13 个入参 `candidate_topk_index` 的形状
  `4096,1,2048`（`indexer_attrs.txt` Input Shapes 末项），也是 config 的 `candidate_topk_blocks`【数】。
- `4065` / `2033` 是 `get_index_k()` 返回的 PA_ND buffer 形状 `[pages, page_size=64, 1, 128]` 的 dim0【数】。
  它由 `POOL:277` `npu_num_pages = (self.size + kp + 1)//kp` 决定，`size` 是该 ratio 的 token 池容量：
  4065×64≈260 160、2033×64≈130 112，正好 **2:1** —— 因为 ratio-1 每 token 一行、ratio-2 每 2 token 一行，
  两者池容量（≈260K token 上下文）相同（`POOL:643-658` 构造，注释：`Ratios 1 and 2 share the same packed-fp4 row shape,
  told apart only by size`）。**所以"2033 → 4065"是 ratio-2 池 vs ratio-1 池，不是候选数翻倍。**

候选机制的代码作用点：

| 名字 | 代码位置 | 作用 |
|---|---|---|
| `candidate_source_layer_id=20` | `IDX:225`；`BACKEND:2867` 读入 | 决定哪一层当"一级发布者" |
| `candidate_topk_blocks=2048` | `IDX:227`；`BACKEND:1599-1606`（LIV2 输出）、`BACKEND:1740-1761`（SLI 输入） | 一级最多发 2048 个块；二级把 `candidate_topk_index` 末维当块数 |
| `candidate_block_size=8` | `IDX:228`；`BACKEND:1606, 1762` | 每个逻辑块 = 8 个压缩 token，由算子在内部展开（`BACKEND:1731-1734` docstring） |
| `candidate_block_indices` | 生产：`BACKEND:1599-1608`（LIV2 第 3/4 返回值）→ `BACKEND:1441`（存 `fm.dsv41_candidate_blocks`）；消费：`BACKEND:1740-1746` → 实参 `BACKEND:1760` | 块号（int32，-1 填充），行数必须 == T（`BACKEND:1747-1752` 校验） |
| `candidate_topk_blocks` 上限 | `OP_SLI/op_host/sparse_lightning_indexer_tiling.cpp:521-527`（`(0,2048]` 且 64 的倍数）；`OP_LIV2/.../lightning_indexer_v2_tiling.cpp:345` 同规 | 两级都只接受 ≤2048 且 64 倍数 |

**两级结构确实存在，但不是"先粗筛再精算省算力"**——这是本次最重要的语义发现：

`OP_SLI/README.md`（功能说明 + 公式）写得很明确：候选**内部**保留原分，候选**外部但可达**的位置被降级成 `NEG_HUGE(-1e30)`
**继续参与排序**（"leak 降级"，仍可作填充入选），不可达位置才 `-inf`。也就是说 **SLI 仍然对整条压缩历史做完整打分**
（`s2Size` 来自 block_table，见 Q4），候选集只影响"排序时的降级"，**不减少任何矩阵乘**。

**`SparseLightningIndexerMetadata` 不是这个角色**：本次 trace 里根本不存在这个算子（`op_statistic.csv` 无此项）。
存在的 metadata 算子是 `SparseFlashMlaMetadata`（3 次/forward，AI_CPU，`BACKEND:2074`），服务的是 attention 而不是 indexer。
A3 上"一级"就是 `LightningIndexerV2` 的 candidate 输出（`BACKEND:1599` 的 `candidate_topk_blocks=2048`），
不存在独立 metadata kernel；A5 上才有 `quant_sparse_lightning_indexer_metadata_dsl`（`BACKEND:1934`）这个独立 dispatch-table 算子。

### Q4. `Block Num=20` 由什么决定？Python 能改吗？

**结论：由 tiling host 用"平台核数"算出来，与 shape / tile / workspace 全都无关；Python 侧无法影响。**

`OP_SLI/op_host/sparse_lightning_indexer_tiling.cpp:1084-1090`：

```cpp
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);
```

- 对 MIX 算子：`SetBlockDim = 20`（= `GetCoreNumAic()`），AIV 跟随 40 → profiler 显示 **Block Num=20 / Mix Block Num=40**。
- 反证（同一份 trace）：`OP_LIV2`、`SparseFlashMla`、甚至 `BatchMatMulV2` 全是 20/40；而纯 AIV 的
  `CamMoeCombineNormal` 是 **40/0** —— 它的 tiling 是 `cam_moe_combine_normal_tiling.cc:589`
  `CalcTschBlockDim(aivNum, 0, aivNum)` ⇒ 40。**同一个 `CalcTschBlockDim`，只因为传了 `aicNum` 与否，输出 20 或 40**
  ⇒ Block Num 就是"这颗 die 上参与该 kernel 的 AI core 数"，不是算法 tiling 常量。
- workspace 是**反过来**由 `aicNum` 算出来的（`tiling.cpp:1104-1114`：`mm1ResSize*...*aicNum`），所以"调 workspace 影响 block 数"不成立。
- 规模无关性：T=4096 行固定切成 20 份（每核 ~205 行），S2 在核内循环（`OP_SLI/op_kernel/arch22/sparse_lightning_indexer_kernel_arch22.h:330`
  注释 "D4: consumer 恒禁 S2 跨核切分"，所以 S2 不能靠加核切）。**S2 越长，单核串行越长，网格不变。**

**Python 侧能改的边界**：
- 改不了 `blockDim`（没有任何 Python 参数进到 `GetCoreNumAic`）。
- 能改的只有"喂给 tiling 的 shape"：`T`（chunk 大小 / `--chunked-prefill-size`）、`S2`（上下文/前缀命中）、
  `candidate_topk_blocks`（config）、`candidate_block_size`（config）、`topk`（`index_topk=512`）、`n_heads/head_dim`（模型结构）。
  这些只改变"每核干多少活"，不改变 "20 个核"。
- **如果 kernel 实现在预编译 wheel（`sgl_kernel_npu` / CANN ops-transformer）里，Python 侧的边界就到"传什么 shape/attr"为止**：
  tiling 里怎么切 S1/S2、候选掩码怎么展开、每 tile 多少标量操作，只能在算子包里改并重新出包。
  本次可复核的算子源码在 `/Users/an/Developer/ops-transformer`（9.2.0 快照），**机器上 wheel 的实际内容未验证**。

### Q5. `LightningIndexerV2`(28 ms) vs `SparseLightningIndexer`(984 ms)：差在哪？能走快的那条吗？

**是不同的算子，但不是不同的精度/量化模式**——两者 ABI 几乎一样（q/k/w 同 dtype、同 topk、同 mask_mode=3、
同 `cmp_ratio`、arch22 BF16，`OP_SLI` 是 "cloned from lightning_indexer_v2 @Phase1"），差别只有两处：

1. **多了候选掩码（S1）**：每个 tile 要把"块级候选表"展开成 512 位置的位置级 0/1 掩码，再做 leak 降级/pen 链（`OP_SLI/.../sparse_lightning_indexer_service_vector_arch22.h:326-345`）。
2. **不许 S2 跨核切分**：`..._kernel_arch22.h:330` 注释 "D4: consumer 恒禁 S2 跨核切分（候选行掩码/pen 链在单核内完成，免 LD 归并）"，
   即 SLI 只能按 S1 切核、S2 全在核内串行；LIV2 侧是否有跨核 S2 未逐行核对（**未验证**）。
   但这不构成 35× 的原因：不管怎么切，两者要算的 MAC 总数一样（Q4 的满速估算已经含全量）。

**984 ms 的机制（有算子源码里的实测标注，可复核）**——`sparse_lightning_indexer_service_vector_arch22.h:326-334` 原文：

```
// 【2026-09-22 性能重写（P1）】原实现（2026-09-21 缺陷修复轮）每 tile 每行做
//   512 次 SetValue 初始化 + candBlocks 次标量 GetValue 全量扫描 + hits×blockSize 次 SetValue 置位
//   + 一对 S_V fence；候选集合对整行所有 tile 不变，扫描结果本质重复。实测（mask_sens.py）该路径占
//   consumer 运行时 ~95%：模型主场景 S1/S2c=4096、Kc=2048、bs=8 下 30.1ms vs off 2.0ms（×15.0），
//   且耗时与 candBlocks、S1 均线性相关、与 N1 无关。
```

即：**旧实现的候选掩码是 `O(#tiles × candBlocks)` 的标量重扫**（每 512 个位置就把 2048 个候选块号重读一遍）。

把这段实测标度推到本次 shape（都是【推】）：

| 量 | 值 | 来源 |
|---|---|---|
| S2（SLI） | 1920 × 64 = **122 880** | `indexer_attrs.txt` block_table `1,1920`；tiling `s2 = block_table.dim1 × k.dim1`（`tiling.cpp:804-810`） |
| S2（LIV2 抽样，ratio 2） | 960 × 64 = **61 440** | `kernel_attrs.txt` block_table `1,960` |
| tile 数（512/块） | 240（SLI）/ 120（LIV2） | 512 = `S2_BASE_SIZE`（`tiling.cpp:1101`） |
| 旧掩码路径外推 | 30.1 ms @S2=4096 ⇒ ×30（S2=122 880/4096）≈ **900 ms** | 与实测 **984 ms** 差 9% |
| LIV2 一致性校验 | 2.0 ms @S2=4096 ⇒ ×15（61 440/4096）≈ 30 ms | 与实测 **28.0 ms** 差 7% |

⇒ 两次外推都对上，**984 ms ≈ 旧（legacy）候选掩码路径**。而 `aiv_vec_ratio=6.5% / aic_mac_ratio=1.6% /
aic_scalar_ratio=3.3% / mte2=1.2%` 全部趴着，正说明"cube 一分钟能算完，剩下时间在等掩码链"：
MAC 总量 4096×122 880×32×128 = **2.06 TMAC**（≈4.1 TFLOP），按 20 AIC 满速约 **14-16 ms**
——恰好等于 `aic_mac_time=15.5 ms`。**所以"算得快的那部分只要 16 ms，剩下 968 ms 全是掩码/编排"。**

**能不能走快的那条？**
- 语义上：**不能把 SLI 换成 LIV2 而不改结果**。SLI 的 leak 语义要求"候选外的可达位置被降级但仍参与排序"，
  所以它必须扫全 S2 —— 这一点换任何实现都省不掉（省不掉的是 ~16-80 ms 的 cube 时间，不是 968 ms）。
- 工程上：**慢的部分（968 ms）是已知的实现缺陷，上游算子源码里已经有修复版**（同一文件 `:337-360`：
  "块级 isOut 表每行仅构建一次 + Brcb 向量展开"，把 `O(tiles×candBlocks)` 降到 `O(candBlocks + tiles)`）。
  如果机器上装的 ops-transformer 是旧于 2026-09-22 的版本，升级算子包即可（**Python 侧一行都不用改**）。
- 精度/ratio 不是卡点：`ratio=1` 只是让 S2 = T（不被压缩），而 layer 20 的 LIV2 在**同样 S2=122 880、同样 ratio=1、
  还要多做候选输出**的情况下只要 **81.5 ms**（`op_statistic` 里 LIV2 的 max=81 490 µs，且 3×min+max 正好等于 total，
  证明 4 次 = 3×28.0 + 1×81.5，后者就是 layer 20）。⇒ 同层同 shape，"发布候选"几乎免费，
  "消费候选"贵 12×，差价 900 ms 只能是掩码链。

### Q6. `CamMoeCombineNormal` 是哪段代码？为什么 1.7–5.9 ms 波动？

**是 DeepEP 的 combine**（不是 HCCL all-reduce，也不是 `ascend_ops` 的 MC2 那条）：

```
DISPATCH:956        buffer.low_latency_combine(x=..., topk_idx=..., topk_weights=..., handle=...)
 └ deep_ep/deep_ep/buffer.py:716  Buffer.low_latency_combine
    └ deep_ep/strategies/low_latency_strategy.py:126,150  DefaultLowLatencyCommStrategy → self.runtime.low_latency_combine（C++）
       └ sgl-kernel-npu/csrc/deepep/ops2/...   → aclnnCamMoeCombineNormal（OP_ADD 见 op_host/cam_moe_combine_normal.cpp:85）
```

旁证：trace 里同时有 `CamMoeDispatchNormal` / `NotifyDispatch` / `DispatchLayout` 三个 ops2 配套 kernel，
以及 `comm_summary.csv` 的 `distinct_ops` 值（0–4，都是 HCCL allreduce/allgather），
而 `CamMoe*` 系列不在任何 HCCL 汇总行里（它们是 DeepEP 自己的 dispatch/combine kernel）【数+推】。如果是 `--moe-a2a-backend ascend_ops` 会走
`torch_npu.npu_moe_distribute_combine_v2`（`token_dispatcher/ascend_ops.py:358`），trace 里不会出现 `CamMoe*`。

**为什么波动大（0.083 → 5.93 ms，70×）—— 它是"等人"，不是"干活"**：
- 硬件计数：`Block Num=40`（纯 AIV，`aicore_time=0`，说明没有 cube 活），`aiv_mte2_ratio` 0.49–0.72（数据搬运为主）。
- **时长与本地工作量无关**（决定性证据，【数】）：

| kernel_attrs 里的本次输入 | 本地 token 数 | 时长 |
|---|---|---|
| `44,5120;132;384;1024,6;1024,6;1` | **44** | **4 878 µs** |
| `7167,5120;21501;384;1024,6;1024,6;1` | **7167** | **3 100 µs** |

  44 token 比 7167 token 还慢 ⇒ 时间花在"等其他 EP rank 把专家算完并回包"，不是花在自己身上。
- 因此波动 = **层间/rank 间 MoE 负载不均衡 + 跨 die 同步**。C=1、EP 下每个 rank 拿到的 token 数逐层抖动，
  combine 的完成时刻 = 最慢那个 peer；`NotifyDispatch`（avg 612 µs，几乎不动）是它的配对同步点。
- 补充：`op_statistic` 里 `min=82.96 µs`（哪一次对应哪个 batch 没有单独证据，**推断**为近空批/收尾），
  说明它的基线低到几十 µs 级，弹性来自同步而不是本地计算，
  典型搬运型 kernel 的形态。**它不是可优化的算法热点（70.9 ms/chunk，1.3%），但它会把别人的慢放大成自己的墙钟等待。**

---

## 3. 可改 / 不可改的边界

### 3.1 只改 Python 或启动参数就能动的（列参数名）

| 想动什么 | 旋钮 | 现状/依据 |
|---|---|---|
| 两级 indexer 结构本身 | `candidate_source_layer_id`（config.json，`IDX:225-226`、`BACKEND:2867`）；置 `-1` 即关闭两级，8 层全走 LIV2 | 语义变更，需精度评估；但这是唯一能把 4×984 ms 变 4×~80 ms 的 Python 级开关 |
| 候选块数 | `candidate_topk_blocks`（config，`IDX:227`）；取值 `(0,2048]` 且 64 的倍数（`OP_SLI tiling.cpp:521-527`、`OP_LIV2 tiling.cpp:345`） | 只在旧掩码实现下近似线性省时；新实现下收益小 |
| 候选块粒度 | `candidate_block_size`（config，2 的幂 ∈[2,64]，`OP_SLI tiling.cpp:348-356`） | 改覆盖率/语义；对旧掩码路径的 per-tile 成本无影响 |
| 每层 topk | `index_topk=512`（config → `BACKEND:1635` / `1778` 的 `topk`） | 影响 topk 归并，非主因；改小会改精度 |
| chunk 大小 | `--chunked-prefill-size` / 调度侧 chunk 划分 | **对 SLI 无效**：代价 ∝ T×S2，砍 T 只是把一次拆成两次，总量不变（S2 不变） |
| 前缀命中率 | 前缀缓存相关开关（已有 50%） | S2 直接线性下降，是**乘性**收益，但对 128K 场景上限有限 |
| MoE combine 等待 | `--deepep-mode`、`DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ` 等（`_tmp_preflight/run_dsv41_w4a8.sh:185-187`） | 只能缓解，不能消除 EP 的层间不均衡 |
| mHC 的 sinkhorn 次数 | `hc_sinkhorn_iters=20`（config → `MHC_T:33`） | 纯 Python 实现里 39 次 `sum/div`/层/子层；改小是**改语义** |

### 3.2 必须改算子包/kernel 的（依据）

| 项 | 为什么 Python 改不了 | 依据 |
|---|---|---|
| SLI 的候选掩码成本（本轮 968 ms/次 的主体） | 每 tile 的标量重扫写在 kernel 里；修复版是同一文件的"块级表 + Brcb 展开"（2026-09-22 P1 重写） | `OP_SLI/.../sparse_lightning_indexer_service_vector_arch22.h:326-460` |
| `Block Num=20` / S2 不能跨核切 | 由 `CalcTschBlockDim(GetCoreNumAic/Aiv)` + kernel 的 D4 约束决定 | `OP_SLI tiling.cpp:1084-1090`；`..._kernel_arch22.h:330` |
| SLI 必须扫全 S2 | leak 语义（候选外可达位置降级但仍参与排序） | `OP_SLI/README.md` 公式 + `tiling.cpp:804-810`（s2 = block_table×page） |
| mHC 的 fp32 sinkhorn（每 chunk ~0.15–0.3 s，随 T 线性） | A3 走 `mix_stats_torch` 纯 torch 回退；融合版只给 A5（`mhc_pre_dpskv4`） | `MHC:386-389`、`MHC:578-583`、`MHC:685-691`；`MHC_T:14-36` |
| MoE combine 的等待 | 是 EP 通信语义，不是本地 kernel 参数 | `DISPATCH:956` + `cam_moe_combine_normal.cpp` |

---

## 4. 下一步实验建议（便宜 + 能证伪）

### E1（最便宜，纯 config，不改代码）关闭两级 indexer，看 4 个 SLI 是否整体消失

- 做法：`config.json` 里 `candidate_source_layer_id: 20 → -1`（`IDX:225-226` 会让 source/consumer 同时失效），
  重跑同一个 128K 冷启动 trace，只看 `op_statistic`。
- **判据**：`SparseLightningIndexer` 计数变 0、`LightningIndexerV2` 计数变 8、indexer 总时间从 ~4.1 s 降到
  **0.4–0.5 s**（3×28 + 5×~81.5，5 = layer 20/24/28/32/36 都变回 ratio-1 全量 LIV2）。若 indexer 总时间仍在 4 s 量级 ⇒ 我对"SLI 是 4 s 主体"的归因错了。
- 代价：一次冷启动；顺带能测 gsm8k 看精度掉多少（这就是"两级结构值不值 3.6 s/chunk"的直接答案）。

### E2（最便宜，纯 config）`candidate_topk_blocks: 2048 → 512`，区分"旧掩码实现"还是"新实现"

- 做法：只改 config 这一个数（`(0,2048]` 且 64 倍数，两级算子都接受），重跑同一 trace。
- **判据（关键，能直接定版本）**：
  - 若 SLI 单次从 984 ms 降到 **~250 ms（≈1/4）** ⇒ 印证"per-tile 重扫 candBlocks"的旧实现（Q5 成立），
    修复方向 = 升级 ops-transformer 算子包。
  - 若只降到 ~850–950 ms（<10%）⇒ 说明机器上的算子是 P1 重写后的版本，**Q5 的 968 ms 归因被证伪**，
    需要重新找（下一步应该挂 aclprof 的 aicore 细项或看 `aic_scalar_time` 的绝对量）。
- 代价：一次冷启动；注意这条会改精度（覆盖的候选位置少 4×），只作归因，不作上线。

### E3（不占机器）核对机器上算子包的版本是否含 P1 重写

- 做法：在 NPU 机器上只读地翻 `$ASCEND_HOME/opp`（或 wheel 安装目录）里 `SparseLightningIndexer` 的
  tiling/kernel 产物，查是否含 `Brcb` 展开 / `BuildCandBlockTable` 这类符号或 2026-09-22 之后的构建戳；
  与本地 `/Users/an/Developer/ops-transformer`（9.2.0 快照，已含重写）对比。
- **判据**：若机器上是旧构建 ⇒ E2 之前就能定论"升级算子包"。量级估算（【推】）：单次 ~984 ms → ~85 ms
  ⇒ 整轮 indexer 从 67 s（HANDOFF §21.3：144 次 / 465 ms 均值）降到 **~10–12 s**，attention 桶 74.3 s → **~20 s**，
  TTFT 从 73.7 s 落到 **~20 s** 一档（仍余 ~20 s 在等待上）。若已是新构建 ⇒ 立刻停掉这条线，
  转去看 attention 之外的那 30 s（HANDOFF §21 里"12 个 die 在等"的 allreduce 长块）。

---

## 附录 A：数字核对（谁都能重算）

1. **4 次 SLI = 4.00 s**：`SparseLightningIndexer` count=8（2 forward）、total=8 004 696 µs → 每次 1 000.6 ms，每 forward 4 次。
2. **LIV2 = 3×28.02 + 1×81.49**：count=8、total=331 300 µs、min=28 022、max=81 490。
   `3×28 022 + 81 490 = 165 556`，×2 = 331 112 ≈ 331 300（差 0.06%，剩下的来自另一次非严格 min）。
3. **MLA = 148.6 ms/forward**：count=80、total=297 286 µs、min=212.9、avg=3 716。
   38×3.72 ms + 2×0.21 ms ≈ 141.8 + 0.4 = 142 ms（差额为 metadata/波动）。
4. **S2 = 122 880**：block_table `1,1920` × k page 64；LIV2(ratio2) = 960×64 = 61 440（`indexer_attrs.txt` / `kernel_attrs.txt`）。
5. **MAC = 2.06 TMAC**：T=4096 × S2=122 880 × n_heads=32 × head_dim=128 = 2.06e12；对应 `aic_mac_time=15.5 ms`。
6. **4065 / 2033 = 池页数，2:1**：`(size+65)//64`；4065×64≈260 160，2033×64≈130 112。
7. **sinkhorn 次数**：`range(sinkhorn_iters-1)=19` × (2 次 `sum`+`div`) + 1 次初始列归一 = 39 `ReduceSum` +
   39 `RealDiv` + 39 `Adds`/调用；×2 子层/层 ×40 层 = **3120**，实测 3282 / 3200（95%/97.5%）。
8. **每 forward 元素级合计**：Cast 149.1 + Add 91.0 + ReduceSum 84.9 + Mul 77.4 + RealDiv 41.0 + Pows 49.1 +
   ReduceMean 31.9 ≈ **524 ms**（约占 5.43 s 的 9.6%）。

## 附录 B：本次标为「未验证」的项

1. **机器上 `sgl_kernel_npu` / CANN ops-transformer wheel 的实际内容**（是否含 2026-09-22 的 P1 掩码重写）。
   本机只能看到 `/Users/an/Developer/ops-transformer` 的 9.2.0 快照；wheel 版本号 2026.11.0.post4 与源码日期对不上，
   不能据此判定。⇒ E2/E3 就是为这条设的。
2. **`aic_*_ratio` 的统计口径**：我按"该 pipe 忙的墙钟 / kernel 墙钟，取单核口径"来用（这样 `aic_mac_time≈理想 MAC 时间`
   才能对上）。若 profiler 实际是"跨核累加"，则第 5 条的数量级校验要除以 20，结论（98% 空转）仍成立但不再是"满速"。
3. **`Mix Block Num=40` 与 AIV 核数的严格对应**：由 `CalcTschBlockDim` 的语义 + 全 trace 一致模式推断，
   未在 CANN 头文件里逐行核对 `CalcTschBlockDim` 实现。
4. **`indexer_attrs.txt` 里形状字段与入参位置的逐位对应**（我按 A3 `sparse_indexer(q,k,w,topk,cand,block_size,**kwargs)` 对齐，
   `1,1920` 判为 block_table）。数值上 1920×64 = 122 880 与 `index_topk`/窗口都对得上，属强一致，但仍是【推】。
5. **`--chunked-prefill-size` 改小是否真的对 SLI 无收益**：由"代价 ∝ T×S2、S2 不随 chunk 变"推出；
   若实现里对 T 有额外的固定开销（每核 20 份的边界），会略偏。未实测。
6. **4 次 SLI 分别属于 layer 24/28/32/36**：由 `uses_candidates` 定义（layer > 20）唯一确定，属【码】；
   但 trace 里没有 per-kernel 的 layer id，若要逐层对号需要带 stack 的 trace。
