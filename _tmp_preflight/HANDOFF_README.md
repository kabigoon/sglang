# DSV4.1 索引器分析 —— 交接指南（给下一个 agent）

> **你在哪**：sglang 仓库的一个**自用分支**（不是上游分支），里面**没有任何 `python/sglang/**` 的改动**，
> 全部内容都在 `_tmp_preflight/` 下：分析文档 + 实测数据 + 微基准脚本。
>
> **这份文件是入口**。读完它（约 10 分钟）你应该能做到：知道结论是什么、证据在哪、数据怎么读、
> 哪些话不能再讲、用户现在想聊到哪一步。
>
> 📌 **同名文件辨析**：`_tmp_preflight/` 下还有一个更早的 `HANDOFF.md`，
> 那是**上一阶段的整体交接**（任务目标、达标线、整体 TIMELINE、其它算子），仍然有效、可以当背景读；
> 本文件（`HANDOFF_README.md`）是**这一阶段"索引器分析"的专用入口**，两者不冲突。
> 但如果两份文件在"索引器"上有冲突，**以本文件 + `INDEXER_VS_VLLM_DIFF.md` 为准**。

---

## 0. 30 秒上手

| 顺序 | 读什么 | 时间 | 读完能干什么 |
|---|---|---|---|
| 1 | 本文件 | 10 min | 接上整条主线 |
| 2 | `_tmp_preflight/INDEXER_CAP_FINDING.md` | 5 min | 掌握 P0 根因（1 页，带源码坐标 + 实测表） |
| 3 | `_tmp_preflight/INDEXER_VS_VLLM_DIFF.md` | 20 min | 代码级对账 A/B/C + 修复方案（**有勘误节，先看 §0.1**） |
| 4 | `_tmp_preflight/INDEXER_T_S_FINDINGS.md` | 5 min | t(S) 两段拟合、拐点、ns/pair 口径 |
| 5 | 需要时 | — | `_tmp_preflight/KERNEL_CODE_MAP.md`（算子↔代码栈实证）、`VLLM_ASCEND_FINDINGS.md`（整体对比） |

⚠️ **用户当前处于"学习模式"**，不是赶交付：他想先彻底搞懂 `SparseLightningIndexer` 这个算子本身
（公式、形状、两级 TopK、DSV4.1 架构）。**不要一上来就推修复方案**，先看 §9。

---

## 1. 一句话现状

DeepSeek-V4.1-Flash 在 A3×16 上的 128K prefill，**sglang TTFT 73.7 s vs vLLM-ascend 27.4 s（20.4×）**，
其中 **67 s 落在 indexer 上**。根因不是算法、不是 dtype，而是
**`SparseLightningIndexer` 的候选掩码在 `S > 22 016` 时静默回退到一条慢 13× 的标量路径**
（块级 isOut 表容量 `CAND_TBL_CAP=2752` 装不下 `S=131 072` 需要的 16 384 块）。
修复方案是**把这张表从 `float` 改成 `1 bit/块`**（UB 从 64 KB/行降到 2 KB/行），预期 **985 ms → ~75 ms**。
次要发现：sglang 的 topk 累加器宽度没按 `topk=512` 缩（4× 归并浪费，vLLM 侧已修）。

---

## 2. 环境与仓库（4 个只读仓库 + 本仓库）

| 仓库 | 本地路径（**上一台机器**） | 分支 @ commit | 用途 |
|---|---|---|---|
| sglang | `/Users/an/Developer/sglang` | `pr-50205-dsv41-a3-deepep` @ `4291b640a5` | 本仓；只有 `_tmp_preflight/` 是自建的 |
| ops-transformer | `/Users/an/Developer/ops-transformer` | `master` @ `23b4626b`（**shallow，只有 1 个 commit，无历史**） | CANN 自定义算子源码（9.2.0 快照）—— **病根所在** |
| sgl-kernel-npu | `/Users/an/Developer/sgl-kernel-npu` | `learn/sparse-flash-mla` @ `f36769e` | sglang 的 NPU 算子 python 封装（**不含** SLI 的 arch22 kernel 源码） |
| vllm-ascend | `/Users/an/Developer/vllm-ascend` | `main` @ `6fb51c912`（`_cann_ops_custom/` 只有 `.gitkeep`，**vendor 树没进 git**） | 对比对象 |
| vllm-ascend（dsv41-perf worktree） | **`/private/tmp/va0300`** | `dsv41-perf` @ `9af24b562` | 真正在跑的分支 + 全部 vLLM 侧实测数据 |

**校验命令**（换机器后先跑一遍，确认基线没漂）：
```bash
git -C /Users/an/Developer/sglang rev-parse --short HEAD          # 4291b640a5（或本分支 HEAD）
git -C /Users/an/Developer/ops-transformer rev-parse --short HEAD # 23b4626b
git -C /private/tmp/va0300 rev-parse --short HEAD                 # 9af24b562
```

### 2.1 换机器的三个坑（重要）

1. **`/private/tmp/va0300` 和 `/private/tmp/va_calls.csv` 是临时目录，新机器上不会有。**
   本分支已经把**必要的 vLLM 侧数据**复制到 `_tmp_preflight/handoff/data_va/`（见 §5），
   新机器上直接用这些副本即可；**不要再去 `/private/tmp` 找**。
   sglang 侧的数据本来就在 `_tmp_preflight/profile_bundle/` 里，跟着 git 走。
2. **vendor 自定义算子源码（`VllmAscendQuantLightningIndexerV2`）不在任何 git 仓库里**
   —— 它只存在于镜像 `quay.io/ascend/vllm-ascend:v0.30.0-a3` 的
   `/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/`。
   上一台机器只抓到了**文件清单**（`_tmp_preflight/out/.../40_custom_ops.txt`、`21_py_only_in_image.txt`），
   **没有抓到内容**。所以 §4.2 里"镜像跑的那份候选实现"是用 **9.2.0 拆分后的同源代码**做代理的，
   标【推】。要闭环必须上机（§7-2）。
3. **`/Users/an/Developer/vllm-ascend` 停在 `main`**，`dsv41-perf` 是另一个 worktree。
   如果你在新机器上只 clone 了 main，会看不到 `dsv41-perf` 的改动（`git worktree list` 先确认）。

### 2.2 行为约束（沿用上一段会话的规则）

- ❌ **不要改 `python/sglang/**`**（只允许新建/修改 `_tmp_preflight/` 下的文件）
- ❌ **不要跑 server / 压测 / profiler，不要"再采一次数据"** —— 数据已经够用（这是硬约束，有 NPU 机也不行，除非用户明确要求）
- ❌ **不要 `git add -A`**；只 add 自己新建的文件
- ✅ 每条结论必须带 **`file:line`** 或 **config 字段**
- ✅ 区分【码】源码 /【数】实测 /【推】推导 /【未验证】需上机

---

## 3. 口径（**已定论，不要重新推导**）

| 量 | 定义 | 值 |
|---|---|---|
| 硬件 | A3×16 die | — |
| 模型 | DeepSeek-V4.1-Flash(w4a8)，40 层 backbone + 3 层 MTP | — |
| 负载 | 128K prompt / C=1 / chunked prefill **4096** / 50% 前缀命中 | — |
| TTFT | sglang **73.7 s** vs vLLM-ascend **27.4 s** | — |
| `T` | 本 chunk 的 query 行数 | 4096（残 chunk 270 / decode 1） |
| `S` | 本次调用扫描的压缩历史长度 = `block_table 宽 × page` = `kv_len / compress_ratio` | ratio-1 层 `S = kv_len` |
| 稀疏度 | `index_topk = 512` ⇒ 512 / 122 880 = **0.42%** | — |
| 调用结构 | **每 step 8 次 indexer 调用**（不是每层）：3 个 ratio-2 组 × 6 层 + 5 个 ratio-1 组 × 4 层 | 8 = 8 个 `index_source_layer_ids` |
| 对齐口径 | 只比 **ratio-1 consumer**：sglang `SLI@S=kv` ↔ vLLM 4 个 `page128` 调用 | — |

**归一化**：一律用 `ns per (q·k) pair = 时长 / (T × S)`，否则残 chunk（T=270）会污染曲线。

---

## 4. 已确认结论（按证据等级，逐条带坐标）

### 4.1 P0 根因：候选掩码的容量阈值回退（【码】+【数】，三条独立测量，**不要再证明**）

```c
constexpr static uint32_t CAND_TBL_CAP = 2752;                       // :157  块级 isOut 表容量（每行分区）
const int32_t tileBlkNum = s2BaseSize_ / blockSize;                  // :568  512/8 = 64
const int32_t blkCnt     = CeilDiv(actS2Size, 512) * tileBlkNum;     // :572-573
if (blkCnt <= CAND_TBL_CAP) { 快路径（BuildCandBlockTable + ExpandCandMaskToPosition） }
else                        { BuildCandidateMaskLegacyTile(...) }     // :576-588 → :508
```
文件：`ops-transformer/experimental/attention/sparse_lightning_indexer/op_kernel/arch22/sparse_lightning_indexer_service_vector_arch22.h`

- **快路径**：块级表每行建一次 + 每 (行,tile) `Adds(64)` + `Brcb(512)` = **2 条向量指令 / 0 标量**（`:401,:419`）
- **legacy**：每 (行,tile) `512 init + 2048 GetValue + 4096 cmp + h×8 写 ≈ 4 676 条标量`（`:544-555`）
- **阈值**：`bs=8` 时 `S ≤ 22 016` 走快，`S > 22 016` 回退（`CAND_TBL_CAP/64 × 512`）

| S | tiles | blkCnt | 路径 | 实测 |
|---|---|---|---|---|
| 20 480 | 40 | 2 560 | 快 | 12.52 ms |
| **22 016** | 43 | **2 752 = CAP** | 快（边界） | **13.39 ms** |
| **22 528** | 44 | 2 816 > CAP | **回退** | **174.80 ms（×13.1）** |
| 122 880 | 240 | 15 360（5.6×CAP） | 回退 | **985.44 ms** |

【数】来源：① 真实 trace `profile_bundle/indexer_calls_cold.csv`（S=122 880 → 4 次中位 **984.66 ms**）；
② 微基准 `bench_sli.py`（985.44 ms，与 trace 差 0.1%，且把拐点夹到 22016/22528）；
③ `INDEXER_CAP_FINDING.md` 的表。

**成本闭环**【推】：`4096 行 × 240 tile × 4 676 标量 × 15 cycle/op ÷ 40 AIV ÷ 1.8 GHz = 0.958 s`
vs 实测 0.985 s（**误差 2.8%**）。⇒ 慢的就是这条标量链。

**13× 是 Amdahl 天花板**：非掩码工作 74.7 ms + 掩码 958 ms；纯掩码快 ~700×，但只能兑现 13×。

### 4.2 与 vLLM 的对比（【数】+【码】+【推】）

| S | vLLM consumer | ns/pair | sglang SLI | ns/pair |
|---|---|---|---|---|
| 4 096 | 3.98 ms | 0.2372 | 3.26 ms | 0.1945 |
| 20 480 | 24.76 ms | 0.2952 | 12.35 ms | 0.1472 |
| 24 576 | 25.32 ms | 0.2516 | **202.74 ms** | **2.0141** ← 拐点 |
| 122 880 | **48.40 ms** | **0.0962** | 984.66 ms | 1.9563 |

- vLLM **全程无拐点**（相邻 chunk 最大跳变 ×1.023），边际 0.056 ns/pair。
- 差距 = `13.2×（回退） × 1.54×（残差）`。
- **vLLM 为什么不掉坑**【码】：它**没有按块号索引的表**。候选被排成「块号降序 + 下标」的 2048 对数组
  （32 KB，**∝ candBlocks，与 S 无关**），每 tile 用 2 次二分（`CountGE`，`:417-430`）+ 256-lane 距离最小值
  （`:488-494`）+ `Brcb`（`:497`）+ clamp（`:499`）。
  文件：`quant_sparse_lightning_indexer/op_kernel/arch22/quant_sparse_lightning_indexer_service_vector_arch22.h`
- ⚠️ **代理关系**：镜像里跑的是 `VllmAscendQuantLightningIndexerV2`（**拆分前三合一**，有 `candidate_mode 1/2/3`），
  而 9.2.0 快照已按 **拆分提交 P13** 把它拆成 `QuantLightningIndexerV2`(source) + `QuantSparseLightningIndexer`(consumer)。
  所以上面这份 consumer 代码是**同源代码的后续身**，用作代理，标【推】。详见 §4.2.1。

#### 4.2.1 "开源 def 无 candidate"的谜（已解决，不要再当矛盾）
不是快照更新/裁剪，而是**快照已经拆过、镜像还没拆**：
- `quant_sparse_lightning_indexer/README.md:3`："本算子由**拆分提交 P13** 从 `QuantLightningIndexerV2` 的 `candidate_mode=2` 路径独立而来"
- `quant_lightning_indexer_v2/op_host/*_def.cpp:78,94-97`：只有 `candidate_topk_index_out` 输出，**无 candidate 输入**
- `quant_lightning_indexer_v2/docs/aclnnQuantLightningIndexerV2.md:534-537`：第二级"已拆分为独立算子"
- 镜像为什么保留三合一：vLLM 的 `indexer.py:274-288` 是**一次**调用用 `mode` 切 source/consumer

### 4.3 修复方案（C 部分，详见 `INDEXER_VS_VLLM_DIFF.md` §5）

**推荐：块级 isOut 表 `float → 1 bit/块`**
- UB：`S=131 072` 只需 **2 KB/行**（现 float 表 64 KB/行）；`bs∈[2,64]` **全值域**都不再回退（最坏 bs=2 也只要 8 KB/行）
- 展开改造：`Duplicate + And(1<<i) + Mins(1) + Muls(0x3F800000)` 造 64 lane `0.0f/1.0f`，
  再走**原有** staging + `Brcb`（arch22 的 Brcb 只认固定偏移，`:391-394`）⇒ 每 (行,tile) +5 条向量 +2 次标量 ≈ **+0.56 ms**
- **不需要 vote/select**（arch22 无 `Scatter`，`:379`）；**不要**改成"uint8 表"（吃光 UB 余量，bs=2 放不下）
- **不要照抄 vLLM 的排序数组**：按每行扫完 S 归一，vLLM 掩码 `~1.7e6 lane + 1.3e3 标量探针`
  vs sglang 快路径 `1.4e5 lane + 0 标量` ⇒ 它是"内存不随 S 长"，不是"每 tile 更省"
- **预期收益**：985 → **70–80 ms**（三条独立依据：快路径外推 74 ms / 自身 `aiv_vec_time` 硬下界 **66 ms** / 无候选 LIV2 81 ms）
- **残差 1.54×**：向量 66.1 vs 45.3 ms（×1.46）、Cube MAC 16.3 vs 13.7 ms（×1.19）
  ⇒ **BF16→INT8 只值 2.6 ms（3.5%），不值得追**

### 4.4 新增发现（2026-11-28，来自模型 config，**未验证**）

1. **`index_topk = 512`（不是 2048）**。之前文档里"2048 个输出位置"的说法是错的（2048 是 `candidate_topk_blocks`）。
   ⇒ 输出 `[T,1,512]`、稀疏度 0.42%、候选是 **32×** topk（不是 8×）。
2. **sglang 的 topk 累加器宽度没按 topk 缩**：
   ```c
   constexpr uint32_t BASE_TOPK = 2048;                                        // :38
   virTopK = isSparseCountOver2K ? sparseCount : BASE_TOPK;                    // :181  → sparseCount=512 时仍是 2048
   ```
   ⇒ `MergeSort` 目标宽度、`sortOutBuf_`（32 KB）、`Extract` 全部按 2048 维护，只导出 512。
   **vLLM 9.2.0 已经修了**（`topkPairWidth_`，注释 P3，`:222-223,683`）。
   【推】这很可能是 §4.3 里那 1.46× 向量残差的**主要来源**，且**改动比 bitmask 更小**。
   ⇒ 建议单独立项、单独量收益，**不要和 bitmask 混在一个 patch**。

### 4.5 DSV4.1 架构事实（【码】config + 代码，用于解释"为什么这么设计"）

| 事实 | 证据 |
|---|---|
| **40 层只有 4 份 KV**（跨层共享）：ratio-2 → 3 组×6 层；ratio-1 → **1 组×20 层** | `kv_source_layer_ids=[2,8,14,20]`；`kv = compress_pool.kv_buffer[item.compress_layer_id]`（**用 source 层号索引**）`dsv4_memory_pool.py:860-864`；`owns_k = layer_id in config.kv_source_layer_ids` `dsv41_sparse.py:224` |
| 三种层：ratio 0 = 纯 SWA（`sliding_window=128`，无压缩 KV）/ ratio 1 = 全量 latent / ratio 2 = 2:1 压缩 | `compress_ratios`、`get_compress_buffer` 对 ratio 0 返回 `None` `:857-859` |
| indexer 也是"每组一个"：**每 step 8 次**（38 个 ratio>0 的层 → 8 个 indexer） | `index_source_layer_ids` 8 个；`_latest_source()` `model.py:603-628`；`setattr(fm, f"c{ratio}_topk_indices", topk)` `ascend_dsv4_backend.py:1446` |
| 两级 TopK：`candidate_source_layer_id=20` 发 2048 个**块**（×8=16 384 位置），ratio-1 的后续 4 个 indexer 组复用 | config；`fm.dsv41_candidate_blocks` `:1436` |
| 候选外**降级而非过滤**（leak 语义），且**候选一律优先于分数** | `sparse_lightning_indexer/README.md` 公式 + 约束第 7 条（有 `regime 2` 测试）；vLLM 侧同语义 `:502-504` |
| 跨层共享 KV 之所以可行：MLA 吸收 —— `q·(cW^UK^T)^T = (qW^UK^T)·c`，缓存只需 latent `c`（512+64=576/token） | MLA 代数恒等；config `head_dim=512, qk_rope_head_dim=64, num_key_value_heads=1` |
| ⚠️ 共享的 index K 也按 `compress_layer_id` 索引 ⇒ 5 个 ratio-1 组打分用的是**同一份 K**（只有 Q 不同）⇒ 块级候选跨组迁移是**合理外推**，不是盲信 | `dsv4_memory_pool.py:864` |
| 疑点：`index_k_buffer` 分配了 `layer_num`（≈40）份，但只用 4 个键（`compress_layer_id`）⇒ 大量白分配 | `dsv4_memory_pool.py:285-295` vs `:864`（【推】需向 sglang 侧确认；本轮不允许改 python） |

**其它值得知道的架构特性**（只到"config 字段 + trace 里的 kernel 名"级别，未深挖）：
`mHC`（hyper-connections，`hc_mult=4`/`hc_sinkhorn_iters=20`，`HcPre/HcPost` 占 trace 13% kernel 时间）、
`Engram`（`engram_layer_ids=[1,14]`，每层 384 M 条 embedding，`_engram_host_uva_gather_dequant_kernel` = host UVA gather）、
`MTP×3 + DSpark`（`num_nextn_predict_layers=3`、`dspark_block_size=5`、`dspark_markov_rank=256`、`dspark_target_layer_ids=[37,38,39]`）、
MoE 384 routed/6 active、vision tower、YaRN×16 → 1 M 上下文。

---

## 5. 数据字典

### 5.1 sglang 侧（在 `_tmp_preflight/profile_bundle/`，随 git）

| 文件 | 行数 | 关键列 | 能回答什么 | 注意 |
|---|---|---|---|---|
| `indexer_calls_cold.csv` | 288 | `Name, Start Time, Duration, Input Shapes, T, S(derived), Block Num, Mix Block Num, aic_mac_time/ratio, cube_utilization` | SLI/LIV2 逐次调用、t(S) 曲线、拐点 | 含 decode（T=12）；`S(derived)=block_table_w×64`；`Block Num=20 / Mix=40` = 20 AIC + 40 AIV |
| `profile_bundle/indexer_calls_ep.csv` | 280 | 同上 | EP 部署复现 | 与 cold 逐点重合 |
| `prof_128k_cold/kernel_top.csv` | 100 行 SLI | `Duration, aicore_time, aic_mac_ratio, aic_mte1/mte2/fixpipe, aiv_vec_ratio, cube_utilization` | **pipe 计数**（§4.1 的 18% busy 论证） | 只含 top-N 慢调用，**没有 `aic_scalar_ratio` 列** |
| `prof_128k_cold/kernel_schema.txt` | — | 46 列说明 | 原始 kernel_details 的列定义 | 原始 49 MB CSV **没进仓** |

### 5.2 vLLM 侧（**已复制到 `_tmp_preflight/handoff/data_va/`**，原路径在 `/private/tmp`）

| 文件 | 行数 | 关键列 | 能回答什么 |
|---|---|---|---|
| `va_calls.csv` | 6 704（其中 indexer **256** 行） | `bucket, Name, Duration(us), Block Num, Input Shapes, Start Time(us)` | 逐次调用；**按 8 行一组 = 1 个 chunk**（3×page64 + 1×leading + 4×page128）；形状串末段 `4096,1,2048` = candidate 输入 |
| `op_statistic_top.csv` | 6 行 | `OP Type, Count, Total/Min/Avg/Max Time` | **`VllmAscendQuantLightningIndexerV2` Count=384** = 32 prefill chunk × 8 + 16 decode × 8；两个辅助 kernel 各 384 次 |
| `va_buckets.csv` | — | 桶 / kernel 数 / 总时间 | indexer 桶 1248 kernels / 5 892.6 ms |
| `kernel_top.csv` | 176 行 VllmAscend | pipe 比 | vLLM 的 `aic_mac_ratio=0.277 / aiv_vec_ratio=0.915`（对照 sglang 0.016 / 0.065） |
| `va_summary.md` / `meta.txt` / `va_steps.csv` / `step_trace_time.csv` | — | — | 65 步、切步、来源路径 |

**两个数据坑（已踩过）**：
1. `va_calls.csv` 只保留 `Duration > 0.5 ms` ⇒ **384 → 256 行**；缺的 128 行是 decode/廉价调用。
2. vLLM 的 `Input Shapes` 里 **S 是预分配容量**（恒定 54 347，`54347×128` 或 `54347×64`），
   **实时 S 必须按调用时序重建**（`S ≈ (j+1)×4096`）。**不要从形状读 S。**

### 5.3 其它
- `bench_sli.py`：脱离模型的微基准（连续页池，T=4096，扫 S），**拐点表就是它测的**
- `_tmp_preflight/out/`（在 `/private/tmp/va0300`，**未复制**）：镜像 vendor 树的**文件清单**、`dsa_v41` fork-vs-image diff、
  `12_va_probe.txt`（480 行探针：`index_source` / `candidate_source` / `uses_cfilter` 组合分布）

---

## 6. 代码地图（只列"改哪里 / 读哪里"）

### 6.1 `SparseLightningIndexer`（病根）
`ops-transformer/experimental/attention/sparse_lightning_indexer/`

| 位置 | 内容 |
|---|---|
| `op_kernel/arch22/sparse_lightning_indexer_service_vector_arch22.h:157` | `CAND_TBL_CAP=2752` |
| 同上 `:148-170` | `candBuf_` 32 KB 布局（2×2752 表 + 2048 装载 + 64 staging + 576 pad） |
| 同上 `:381-438` | `ExpandCandMaskToPosition`（每 tile；bs==8 走 `Adds(64)`+`Brcb`） |
| 同上 `:444-506` | `BuildCandBlockTable`（每行一次；8 路展开扫描） |
| 同上 `:508-559` | **legacy 逐 tile 标量重扫**（4676 ops/tile） |
| 同上 `:562-589` | 分派 + 阈值 |
| 同上 `:641,707` | 行循环 / `BuildCandidateMask` 调用点 |
| 同上 `:718-742` | leak 降级 pen 链（`NEG_HUGE − p×2^76` 位域构造） |
| `op_kernel/arch22/sparse_lightning_indexer_kernel_arch22.h:395,723` | S2 tile 循环 |
| `op_host/sparse_lightning_indexer_tiling.cpp:1098,1104` | `S2_BASE_SIZE=512` / `S1_BASE_SIZE=8`（kernel 实际 s1Base=4） |
| `README.md` | **公式 + 跨算子契约 7 条（必读）** |

### 6.2 sglang（调用方 / 角色分派）
`python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py`

| 行 | 内容 |
|---|---|
| `1433-1446` | **角色分派**：`is_candidate_source` / `uses_candidates` / else → 三选一 |
| `1446` | `setattr(fm, f"c{ratio}_topk_indices", topk)` ← 组内共享 |
| `1679-1692` | `_dsv41_a3_index_kwargs`：`seqused_k = seq_lens // ratio`（**S 的权威定义**）、`PA_BBND`、`mask_mode=3` |
| `1729-1755` | `_dsv41_index_qsli_topk`（SLI 调用点，`candidate_indices` 从这里进） |
| `1755` | `sparse_indexer(q, k, w, topk, candidate_indices, candidate_block_size=...)` |
| `2206-2216` | 输出被谁消费：`cmp_sparse_indices=topk.view(-1,1,index_topk)` → `SparseFlashMla` |
| `2859` | `self._dsv4_index_topk = hf.index_topk`（=512） |

`python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py:219-232,285-295`：indexer 的 config 字段（`index_n_heads=32`、`index_head_dim=128`、`weights_proj`、`head_weight_scale=128^-0.5×32^-0.5`）

### 6.3 vLLM（对比 + 代理实现）
- `/private/tmp/va0300/vllm_ascend/models/deepseek_v41/indexer.py:274-288`：三合一调用
- `/private/tmp/va0300/vllm_ascend/attention/dsa_v41.py:1238-1244`：`emits_candidates = ratios[src]==ratio`（**sglang 没有这个 ratio 校验**，见 §8）
- `/private/tmp/va0300/vllm_ascend/models/deepseek_v41/model.py:531-533,603-628`：三个 layer-id 列表 + `_latest_source` 分组
- ops-transformer 侧代理实现：`quant_sparse_lightning_indexer/op_kernel/arch22/quant_sparse_lightning_indexer_service_vector_arch22.h:232,417-430,435-505,640-685`

---

## 7. 未验证清单（只有上机能闭环，按重要性）

1. **`aiv_scalar_ratio`**（慢路径预期 >80%）。bundle 里这一列被裁掉了，一条 profiler 即可补上。
2. **镜像那份 vendor 实现**是否真用"排序数组 + CountGE"（§4.2 是代理）：
   ```bash
   strings .../op_api/lib/libcust_opapi.so | grep -iE "candidate|mode" | head -40
   ls .../op_impl/ai_core/tbe/custom_transformer_impl/dynamic/ | grep -i lightning
   # 打开 fused_quant_lightning_indexer_manage.py，看 candidate 分支是否有
   #   CountGE / SortAll(candPairs) / Brcb(posDist)
   ```
3. `Mins(int32)` / `Muls(int32, 0x3F800000)` 在 arch22 上的行为（bitmask 方案依赖它）
4. bitmask 版在 `bs=2, S=131 072` 的实测 UB 占用（纸面 24.8 KB，看编译器对齐）
5. **`virTopK` 改成 `sparseCount`(512) 后的实测收益**（§4.4-2）
6. 修后 `bench_sli.py` 扫 S=4 K…131 K，确认无拐点且 S=122 880 落在 70–80 ms

---

## 8. 黑名单：已被推翻的结论（**不要重犯**）

| ❌ 不要这么说 | ✅ 应该是 |
|---|---|
| "vLLM 没有候选复用" | vLLM **有**候选（`candidate_mode 1/2/3` + `cand=2048x8` 探针实测），它只是**不用表** |
| "prefix token 多了就慢" | **S 超过 22 016 这个硬阈值**就换实现，与"token 多"无因果 |
| "这个算子本身慢 / BF16 慢" | 同一个 kernel 的快路径比自家 LIV2 还快；慢的是容量不够触发的回退 |
| "升级 ops-transformer 就好了" | **P1 重写已经装了**（快路径就是它），缺的是容量 |
| "`cube_utilization=99%` 说明 cube 满/空" | 该计数**没有区分度**（慢 98.99 / 快 92.6 / vLLM 97.58）；有区分度的是 `aic_mac_ratio`(0.016/0.277) 与 `aiv_vec_ratio`(0.065/0.915) |
| "topk 是 2048" | **`index_topk=512`**；2048 是 `candidate_topk_blocks`（块数） |
| "consumer 只对候选打分所以省算力" | 它**仍然全量 S 打分**（实测 49.5 ms vs source 39.1 ms），候选只限制**排序作用域** |
| "15.4 cycle/op 是实测的" | 是**拟合出来的**（由 985 ms 反推），自洽但不是硬件计数 |
| VA `Input Shapes` 里的 S | 那是**预分配容量**（54 347），不是实时 S |
| "sglang 每层都跑 indexer" | **每组跑一次**（38 层 → 8 次） |

### 8.1 一个真实隐患（别忘了）
sglang 的 consumer 判定 `uses_candidates = 0 <= candidate_source_layer_id < layer_id` **不看 ratio**，
候选槽位也是**全局唯一一个**（`fm.dsv41_candidate_blocks`）。现在能对，靠的是
**"所有 ratio-2 层都排在 layer 20 之前"这个层序约定**。若模型改成 ratio-2 在 source 之后，
sglang 会让它们消费 **ratio-1 位置域**的候选，且只有 `shape[0] != t` 的陈旧检查 ⇒ **静默错误**。
vLLM 有 `ratios[src] == ratio` 挡住。【推】

---

## 9. 教学进度（用户的学习线 —— **下一次对话从这里接**）

用户是**边学边分析**，明确说过"想先搞懂算子"。已讲过（按时间顺序）：

| # | 主题 | 关键结论（可直接复述） |
|---|---|---|
| 1 | 算子定位 | 它是**per-query-row 的检索器**：给每行挑出该看的位置，**只要索引不要分数**（`sparse_values` 恒空，两栈都丢弃 values） |
| 2 | 4 个张量 | `q[T,32,128]` / `k[S,1,128]` / `w[T,32]` / `out[T,1,512]`；`T`=本 chunk token 数；`32`=`index_n_heads`（**不是主 attention 的 64**）；`1`=`num_key_value_heads`(MQA)；`S=kv_len/ratio` |
| 3 | 打分公式 | `score[t,s]=Σ_{g=0}^{31} w[t,g]·ReLU(q[t,g,:]·k[s,0,:])`；`g` 是 head 维（G=N1/N2，本算子**硬约束 N2==1**）；kernel 里切成 **2 批 × 16** 累加（`groupInner_=16`，UB 预算决定，不是语义） |
| 4 | 两级 TopK | source 层（config `candidate_source_layer_id=20`）发 **2048 个块**（×8=16 384 位置=13% of S）；consumer 在候选内精排；**候选外降级 −1e30 但仍可填充入选**；排序键是 `(是否候选, 分数)` **字典序** |
| 5 | 为什么有"块" | ①跨层共享必须"欠定"成区域先验 ②块级聚合降噪 ③`bs=8` = 32B 向量粒度（`CAND_BRCB_FANOUT=8`） |
| 6 | 组与共享 | 38 个 ratio>0 的层 → **8 个 indexer**（3×6 ratio-2 + 5×4 ratio-1）；组内共享 topk（per-ratio buffer，下一组覆盖）；ratio-1 组间还共享块级候选 |
| 7 | 设计 vs 实现 | "候选优先"是**模型设计**（config 字段 + topology 校验 + `regime 2` 契约），不是算子图方便；代价：第一级漏=永久漏 ⇒ 所以第一级要宽（32× topk） |
| 8 | KV 共享 | 40 层只有 **4 份 KV**（`kv_buffer[compress_layer_id]`）；靠 **MLA 吸收** 使共享可行 |
| 9 | MLA 吸收 | `q·(cW^UK^T)^T = (qW^UK^T)·c`；缓存只要 latent `c`(512+64=576)；V 侧同理 `out=(Σp·c)W^UV`；V4.1 里 `head_dim=r=512` ⇒ 吸收**不增加点积维度**（V3 里是 +2.7× 算力换 85× 显存） |
| 10 | 符号约定 | `_`=下标，`^`=名字标签（**不是幂**），例外 `^T`=转置；`x_t`=进入该层 attention 的隐藏向量（5120 维，是 mHC `HcPre` 混合 4 条残差流之后的产物） |

**下一步用户可能会挑的**（按他上次的选项）：
1. `vl(b,i)` / `-1` / 因果在**压缩域**怎么对齐（含 `cmp_residual_k`、`mask_mode=3`）
2. `w` 的来源（`weights_proj: hidden→32` + `head_weight_scale=1/√128·1/√32` 为什么乘在 w 上而不是 q 上）
3. **kernel 一个 tile 的 ①–⑦ 流水**（① CopyIn w+cube 结果 → ② DoScale/DoReduce → ③ 造 index → ④ BuildCandidateMask → ⑤ pen 链 → ⑥ SortAll+MergeSort → ⑦ Extract+CopyOut）

**回答风格**：用户习惯用"我的理解是……对吗？"来确认。**先给判定（✅/⚠️/❌），再给精确版，再给一个能手算的小例子**，他反馈很好。

---

## 10. 下一步（工作线，按优先级）

1. **P0**：把 bitmask 方案整理成给算子团队的 issue / patch 草案（diff 骨架 + UT 清单 + 验收口径 985→75 ms）
2. **P0**：`virTopK` → `sparseCount` 的独立 patch（改动更小，可能收益更大，需单独立项量测）
3. **P1**：`INDEXER_VS_VLLM_DIFF.md` 里两处勘误已加（§0.1 / §5.7），可继续补 §4.5 的架构事实
4. **P2**：向量侧调优（`AlignS2`、sort/merge 宽度）—— 上限就是那 1.46×
5. 教学线：按 §9 的 1→2→3 继续

---

## 11. 文件清单（本分支新增/相关的全部内容）

```
_tmp_preflight/
├── HANDOFF_README.md              ← 本文件（入口）
├── INDEXER_VS_VLLM_DIFF.md        ← 代码级对账 A/B/C（含 §0.1 勘误、§5.7 新发现）
├── INDEXER_CAP_FINDING.md         ← P0 根因（1 页）
├── INDEXER_T_S_FINDINGS.md        ← t(S) 曲线 / 拐点 / ns-pair 口径
├── KERNEL_CODE_MAP.md             ← 算子↔代码栈实证（谁发起、什么形状）
├── VLLM_ASCEND_FINDINGS.md        ← 主窗口整体对比（§9 索引器对账）
├── bench_sli.py                   ← 脱离模型的微基准（拐点表来源）
├── profile_bundle/                ← sglang 侧实测数据（逐次调用 + pipe 计数）
└── handoff/data_va/               ← **vLLM 侧数据副本**（原在 /private/tmp，新机器必备）
```

---

## 12. 如果你（下一个 agent）只有 5 分钟

记住这五句就够开工：
1. 慢在 `ops-transformer/.../sparse_lightning_indexer_service_vector_arch22.h:157` 的 `CAND_TBL_CAP=2752`：
   `bs=8` 时 `S>22 016` 就回退到每 tile 重扫 2048 个候选块的标量路径，**慢 13×**。
2. 修法是**块级表改 bitmask**（2 KB/行），不是照抄 vLLM、不是换 dtype。
3. vLLM 快的真正原因是**候选表不随 S 长**（排序数组 + 二分），不是它"没有候选"。
4. 数据都在 `_tmp_preflight/`（vLLM 侧的副本在 `handoff/data_va/`），**不要再采数据**。
5. 用户现在想**先学懂算子**；回答用"先判定、再精确、再举例"，每条结论带 `file:line`。
