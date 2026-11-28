# SparseLightningIndexer vs VllmAscendQuantLightningIndexerV2 —— 代码级对账与修复方案

> 结论一句话：**sglang 的快路径结构并不比 vLLM 差（sglang 每 (行,tile) 只有 2 条向量指令，
> vLLM 一个 2048 位置的 tile 要 ~110 条 256-lane 指令），
> 差的是「块级表按全局块号稠密索引」带来的容量耦合**。修 bitmask 后 SLI ≈ 70–80 ms
> （实测硬下界 66 ms），残差 1.55× 几乎全在向量侧，**BF16→INT8 只剩 2.6 ms（3.5%），不值得追**。
>
> 标记约定：【码】源码（file:line）｜【数】实测（repo 里的 CSV/trace）｜【推】由前两者推导｜【未验证】需上机。

---

## 0. 结论速览

| 问题 | 结论 | 依据 |
|---|---|---|
| 13× 从哪来 | legacy 每 tile 重扫 2048 个候选块（O(tiles×candBlocks)），**单次调用 4.60e9 条标量指令**；判为标量依赖链 ~15 cycle/op 时算出 0.96 s，实测 0.985 s（差 3%） | §2.4–2.5 |
| 为什么是 13× 而不是 700× | 快路径还有 75 ms 的「非掩码」固定工作，Amdahl 把纯掩码的 ~700× 压到 13× | §2.6 |
| 硬件计数吻合吗 | 吻合。慢路径 7 条 pipe busy 之和 = **18.1%**（csv 里 6 条为 14.8%），即 ~82% 纯空转；快路径 `aic_mac_ratio` 0.209、慢路径 0.016；`cube_utilization` 慢 98.9 / 快 92.6 **没有区分度，不能用它做证据** | §2.7 |
| vLLM 为什么不掉坑 | 它不建「按块号索引」的表。候选**按块号降序排成一个 2048 对的数组**（32 KB，**与 S 无关**），每 tile 用 2 次二分 + 256 lane 距离最小值展开（它的 `s2Base=2048`，比 sglang 的 512 粗 4 倍） | §4 |
| sglang 该不该照抄 | **不该。** 照抄只换数据结构、不减少工作量（按每行扫完 S 归一：vLLM 掩码 ~1.7e6 lane + 1.3e3 标量探针，sglang 只要 1.4e5 lane + 0 标量），而 1.46× 的残差在 topk/sort 管线里，跟掩码结构无关 | §4.7 |
| bitmask 可行吗 | 可行。S=131 072 只需 **2 KB/行**（现 float 表 64 KB/行）；且覆盖 `bs∈[2,64]` 全值域（最坏 bs=2 也只要 8 KB/行），**回退路径可以整段删除** | §5.1 |
| Brcb 不能吃 bitmask | 不需要 vote/select。`Duplicate + And(1<<i) + Mins(1) + Muls(0x3F800000)` 把 64 bit 变成 64 个 0.0f/1.0f，再接现有 staging+Brcb；比现在多 5 条 64-lane 向量 + 2 次标量 ≈ 0.56 ms（0.7%） | §5.2–5.3 |
| 改完多少 | **70–80 ms**（三条独立依据：快路径线性外推 74 ms / 自身 aiv_vec_time 硬下界 66 ms / 无候选的 LIV2 81 ms） | §5.4 |
| **勘误（§0.1）** | `index_topk=512`（不是 2048）⇒ 输出 512/行、稀疏度 0.42%、候选=32×topk；另发现 topk 累加器未按 512 缩（4× 归并浪费，vLLM 已修） | §0.1 / §5.7 |
| 剩下的 1.55× 该不该追 | 拆开：向量 66.1 vs 45.3 ms（×1.46）、Cube MAC 16.3 vs 13.7 ms（×1.19）。**INT8 只值 2.6 ms**，不值得；要追就追 topk/sort 的向量常数 | §5.5 |

---

## 0.1 勘误与新增发现（2026-11-28，来自模型 config）

> 用户提供了 `DeepSeek-V4.1-Flash-1017-w4a8/config.json` 的 `text_config`，其中两个字段修正/补充了本文：

### 0.1.1 勘误：`index_topk = 512`（不是 2048）
- 本文早前多处把"每行输出 2048 个位置"当作 topk，**错**。2048 是 `candidate_topk_blocks`（**块数**）。
- 实际：`index_topk: 512`【数】config；`self._dsv4_index_topk = hf.index_topk`【码】`ascend_dsv4_backend.py:2859`。
- 影响：
  - 输出是 `[T, 1, **512**]`；稀疏度 512/122 880 = **0.42%**（不是 1.6%）。
  - 候选 = 2048 块 × 8 = 16 384 位置 = **32× topk**（不是 8×）⇒ 第一级筛得比本文原先描述的更宽。
  - **不影响** §2 的掩码成本账（每 tile 的标量数与 topk 无关）与 §5.4 的 75 ms 预测（实测外推，不含 topk 语义）。

### 0.1.2 新增发现（未验证）：topk 累加器宽度没按 `topk` 缩
```c
constexpr uint32_t BASE_TOPK = 2048;                                   // service_vector_arch22.h:38
virTopK = isSparseCountOver2K ? sparseCount : BASE_TOPK;                // :181
```
`sparseCount = 512 ≤ 2048` ⇒ `isSparseCountOver2K = false`（`kernel_arch22.h:207`）⇒ **`virTopK = 2048`**。
于是 `MergeSort` 目标宽度、`sortOutBuf_`（2×2048×2×4B = 32 KB）、`Extract` 全部按 **2048** 维护，
最后只导出 512（`:777-780` 的 `copyLen = sparseCount`）⇒ **每 tile 归并宽度多 4×**。

vLLM 侧 9.2.0 **已经修了**：`topkPairWidth_`，注释 "P3: 累加器按 topk 宽度分配"【码】
`quant_sparse_lightning_indexer_service_vector_arch22.h:222-223,683`。

【推】这很可能是 §5.5 那 1.46× 向量残差的**主要来源**，且改动比 bitmask 更小。
⇒ 建议**单独立项、单独量收益**，不要和 bitmask 混在一个 patch（详见 §5.7）。

---

## 1. 口径对齐（1 处精化 + 2 处修正）

### 1.1 已确认的口径，直接沿用

A3×16 die；DeepSeek-V4.1-Flash(w4a8)；128K prompt / C=1 / chunked prefill 4096 / 50% 前缀命中。
TTFT sglang 73.7 s vs vLLM-ascend 27.4 s。
`T` = 本 chunk 的 query 行数（=4096，残 chunk 270）；`S` = 本次调用扫描的压缩历史长度 = block_table 宽 × 64。
SLI 只跑在 ratio-1 层 ⇒ `S ≈ kv_len`【码】`ascend_dsv4_backend.py:1755` 只在 `_dsv41_index_qsli_topk` 调用，
而 ratio 由 `_dsv41_a3_index_kwargs(forward_batch, ratio)` 传入【数】`indexer_calls_cold.csv` 中 SLI 的 `S(derived)`
等于同期 ratio-1 的 kv_len（4096…126 976，步长 4096）。

### 1.2 精化：12 次/chunk 里，能用来对账的是 **8 次**

题面的「32 chunk × 12 = 384」是对的【数】`.../k128k/op_statistic_top.csv`
（`VllmAscendQuantLightningIndexerV2, MIX_AIC, Count=384, Total=5 518.3 ms`）⇒ **384 = 32 × 12**，
且 12 次是**同一个算子**（不是别的 kernel 混进来）。
但 `va_calls.csv` 只保留 `Duration > 0.5 ms` 的调用 ⇒ 只剩 **256 = 32 × 8**；另外 4 次/chunk 未落盘。
按形状与顺序，落盘的 8 次**顺序固定**为 `3×page64 → 1×page128 → 4×page128`：

| 组内角色 | k 形状 | 次数/chunk | 时长（chunk0 → chunk30） | 含义 |
|---|---|---|---|---|
| 半维族 | `[54347,64,1,128]` | 3 | 0.69 → 14.79 ms | ratio-2 consumer |
| leading | `[54347,128,1,128]` | 1 | 2.20 → 39.08 ms | ratio-1 candidate **source**（mode=1） |
| 4 个完全相同 | `[54347,128,1,128]` | 4 | 3.98 → 49.51 ms | ratio-1 **consumer**（mode=2） |

【数】`va_calls.csv`（形状枚举 6 种、计数 160/96；形状串末段 `4096,1,2048` 出现即 candidate 输入槽非空）。
那 4 次 <0.5 ms 的调用：从探针的 `uses_cfilter` 分布看，应是 `uses_cfilter=False`（mode=3，不做候选过滤）的变体
【推】`out/fail_.../12_va_probe.txt:10-15`（216 条 cfilter=False vs 144 条 cfilter=True）【未验证】。
另：同桶还有 `_prepare_indexer_indices_kernel` 与 `_quantize_indexer_query_kernel`，
各 **384 次**（与本算子 1:1，avg 626 µs / 295 µs）【数】`op_statistic_top.csv` —— §5.5 用到这两个数。
⇒ 对账只数本算子：**12 次/chunk，其中 8 次可量（3 半维 + 1 leading + 4 consumer）**。

### 1.3 修正一：两侧每 chunk 的层拓扑其实是同一个 3 / 1 / 4

sglang 同 chunk 的 8 次调用（`indexer_calls_cold.csv` 按 `Start Time` 排序）也是 `3 + 1 + 4`：

| 位置 | sglang 算子 | S(derived) | 与 vLLM 的对应 |
|---|---|---|---|
| 1–3 | `LightningIndexerV2` ×3 | `kv/2`（2048, 6144, …） | ≡ 半维族（page 64） |
| 4 | `LightningIndexerV2` ×1 | `kv`（4096, 8192, …） | ≡ leading（ratio-1 source） |
| 5–8 | `SparseLightningIndexer` ×4 | `kv` | ≡ 4 个完全相同的 consumer |

【数】`indexer_calls_cold.csv`：chunk0 = `LIV2@2048 ×3, LIV2@4096 ×1, SLI@4096 ×4`，
chunk1 = `LIV2@4096 ×3, LIV2@8192 ×1, SLI@8192 ×4`，逐 chunk 循环。
⇒ **题目原先担心的「4/4 vs 5/3 层数不一致」不存在**，两边都是 5 个 ratio-1 + 3 个 ratio-2。
⇒ 本文所有 A/B/C 的对比只取 **ratio-1 consumer（sglang SLI@S=kv ↔ vLLM 4 个 @page128）** 这一对。

> 仍存的一处真实差异（不影响本贴结论，列为待确认）：
> sglang 的 3 次 ratio-2 调用是 `LightningIndexerV2`，其 def 里**没有候选输入**（只有 `candidate_topk_index_out` 输出）
> 【码】`experimental/attention/quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_def.cpp:78`；
> vLLM 的 3 次 ratio-2 调用**带 `candidate_topk_index` 输入**（§1.2 表「半维族」行的形状末段非空）【数】。
> 即 ratio-2 层在 vLLM 侧走两级 TopK、在 sglang 侧走全量 TopK。该差异使 ratio-2 的 3 次调用
> sglang 28.02 ms vs vLLM 14.79 ms（S=63488/61440，×1.9），但总量只有 84 ms/chunk，不是主矛盾。

### 1.4 修正二：`va_calls.csv` 与 `dsv41-perf` worktree 的实际路径

题面写的是 `<vllm-ascend>/_tmp_preflight/profile_bundle/k128k/va_calls.csv`，**该文件不在那个目录**
（该目录只有 `kernel_top.csv / meta.txt / op_statistic_top.csv / step_trace_time.csv / va_buckets.csv / va_steps.csv / va_summary.md`）
【数】`ls`。真正的 256 行明细在 **`/private/tmp/va_calls.csv`**。
另外 `dsv41-perf` 分支在 `/Users/an/Developer/vllm-ascend` 是 **main**，该分支的实际 worktree 是
**`/private/tmp/va0300`**（`git worktree list`）【数】。本文所有 vLLM 侧 file:line 都用 `/private/tmp/va0300` 下的路径。

---

## 2. A) 两条路径的复杂度账

### 2.1 循环结构（决定「每行每 tile」的粒度）

```
kernel: for s2LoopIdx = s2Start .. s2LoopEnd            // S2 tile 循环，外层
          ProcessVec(runInfo)                           // kernel_arch22.h:395 / :723
            for innerS1Idx < cuS1ProcNumPerAiv(=≤2 行)   //   向量核内 S1 行循环，内层
              ... DoScale/DoReduce → score 行
              BuildCandidateMask(...)                   //   service_vector_arch22.h:707  ← 每 (行, tile) 一次
```
【码】`op_kernel/arch22/sparse_lightning_indexer_kernel_arch22.h:395,723`；
`op_kernel/arch22/sparse_lightning_indexer_service_vector_arch22.h:641,707`。
并行度：`Block Num=20 / Mix Block Num=40`（20 AIC + 40 AIV）【数】`indexer_calls_cold.csv` 全表恒定 20/40。

```
s2BaseSize_ = 512                                     // tiling.cpp:1098 S2_BASE_SIZE=512
tileBlkNum  = s2BaseSize_ / blockSize  = 64           // :568    (bs=8)
blkCnt      = ceil(actS2Size/512) * tileBlkNum        // :572-573
numTiles    = ceil(S/512)                             // 由 s2 循环界推出
```

### 2.2 快路径公式（`blkCnt ≤ CAND_TBL_CAP`）

每行**一次**（行首 tile，`:576-580` → `BuildCandBlockTable :444-506`）：

```
tblLen = Align(blkCnt, 64)                                       // :450
W_fast_row_build = tblLen×SetValue  +  candBlocks×GetValue        // :458-460 / :480-500
                 + candBlocks×(2 cmp + 命中×SetValue) + 2 fence
```

每 **(行, tile)**（`:583` → `ExpandCandMaskToPosition :381-438`，bs==8 分支 `:396-402`）：

```
W_fast_tile = Adds(stage, tbl[tileBlockBase], 0.0f, 64)           // :401  1 条向量（64 lane）
            + Brcb(candIsOutUb_, stage, 8, {1,64})                // :419  1 条向量（512 lane）
            + PipeBarrier<PIPE_V>
```

⇒ **每行每 tile = 0 条标量指令 + 2 条向量指令**。这是 P1 重写的核心，已装上。

### 2.3 legacy 公式（`blkCnt > CAND_TBL_CAP` → `BuildCandidateMaskLegacyTile :511-559`）

每 **(行, tile)**：

```
p = 0..511                 : 512 × SetValue(CAND_IS_OUT_ONE)                 // :544-546
w = 0..candBlocks-1        : candBlocks × GetValue(candRow[w])                // :548
                             candBlocks × 2 次比较 (-1 脏数据/越界免疫)        // :549
                             命中 h 次 × blockSize × SetValue(0)              // :551-553
1 × V_S fence + 1 × S_V fence                                                  // :543 / :556-558
```

其中命中数（候选均匀分布时的期望）

```
h(S) = candBlocks × tileBlkNum / blkCnt = candBlocks/numTiles = 2048/240 ≈ 8.53
```

⇒ **每行每 tile ≈ 512 + 3×2048 + 8.53×8 = 4 676 条标量指令**（与 tile 序号无关，因为候选集对整行不变）。

### 2.4 代入 S=122 880、T=4096、candBlocks=2048、blockSize=8

| 量 | 快路径（**容量够时**；今日 CAP 下 S=122 880 跑不到，即 bitmask 修复后的形态） | legacy | 倍差 |
|---|---|---|---|
| numTiles | 240 | 240 | — |
| blkCnt | 15 360 | 15 360 | — |
| 每行一次性 | `tblLen=blkCnt=15 360` 次 SetValue + 2048 次 GetValue | — | — |
| **每行每 tile 标量** | **0** | **4 676** | ∞ |
| 每行每 tile 向量 | 2 条（64 + 512 lane） | 0 | — |
| **每行全程标量** | 15 360 + 2048 + 2048 ≈ 1.9e4 | 240 × 4 676 = **1.12e6** | ×58 |
| **整次调用标量**（T=4096 行） | 7.9e7 | **4.60e9** | ×58 |
| 整次调用向量 | 4096×240×2 = 2.0e6 条（64/512 lane） | 0 | — |

【推】由 §2.2/§2.3 公式 + §1.1 口径直接代入。

### 2.5 时间模型 → 3% 误差闭环

用 `bench_sli.py` 的 A/B 对（同 T、相邻 S，只差 1 个 tile，一条走快一条走回退）：

| S | tiles | blkCnt | 路径 | 实测 | 每 tile |
|---|---|---|---|---|---|
| 22 016 | 43 | 2 752 = CAP | 快 | 13.39 ms | 0.3114 ms |
| 22 528 | 44 | 2 816 > CAP | **回退** | **174.80 ms** | **3.973 ms** |

【数】`INDEXER_CAP_FINDING.md` 表；`bench_sli.py` 独立微基准。
（注意 `174.80 − 13.39 = 161.4 ms` 不是「一个 tile 的代价」：它是 44 个 tile **全部**从快切到慢的差额，
44 × (3.973 − 0.311) = 161.1 ms ✓。）

换成「每行每 tile 的 AIV 时间」：`t_rowtile = t_call × 40 AIV / (T × tiles)`

```
快：0.3114 ms × 40 / 4096 = 3.04 µs·AIV
慢：3.973  ms × 40 / 4096 = 38.8 µs·AIV        →  每 tile ×12.76
掩码增量 = 38.8 − 3.04 = 35.8 µs·AIV per row-tile
```

⇒ 隐含标量指令成本 = `35.8 µs / 4 676 ops = 7.66 ns/op ≈ 13.8–15.3 cycle/op @1.8–2.0 GHz`。
这正是「UB 标量读（延迟 ~10–15 c）+ 结果决定分支」的**无 ILP 串行链**成本，与代码形态一一对应
（legacy 的 `for w=0..2047` 是朴素的依赖循环，**没有** `CAND_SCAN_UNROLL` —— 那个 8 路展开只用在了
块级表构建 `:480-500`）。

闭环校验（S=122 880）：

```
预测 t = T × numTiles × 4676 ops × 15 cycle/op / 40 AIV / 1.8 GHz
       = 4096 × 240 × 4676 × 15 / 40 / 1.8e9 = 0.958 s
实测 t = 0.985 s   →  误差 2.8%
```
【推】+【数】`indexer_calls_cold.csv` S=122 880 的 4 次中位 984.66 ms。
（快路径同口径：0.3114×240 = 74.7 ms + 一次性 ≈ 75 ms —— §5.4 引用的就是这条。）

### 2.6 为什么只有 13×，不是 700×

纯掩码比：`35.8 µs / (2 条向量指令 ≈ 0.05 µs) ≈ 700×`。
但整次调用里还压着 75 ms 的非掩码工作（cube + reduce + sort/merge），

```
t_slow / t_fast = (74.7 + 958) / (74.7 + 0.3) = 13.8     实测 13.05–13.2
```

⇒ **13× 是 Amdahl 给出的收益天花板，不是掩码本身的倍率**（掩码本身快 ~700×，但只能兑现 13×）。这解释了为什么「P1 已把掩码做到 0 标量，
掉进回退后却只慢 13×」——非掩码那 75 ms 占了慢路径的 7.2%。
同时也给出一个可执行判据：**只要把容量补上，收益上限就是 13×（985 → 75 ms），不可能再靠优化掩码多拿。**

### 2.7 硬件计数核对：吻合，但 `cube_utilization` 不是证据

同一 AIV 核上各 pipe 的 busy 比【数】`prof_128k_cold/kernel_top.csv`（`aic_mac_ratio / aic_mte1 / aic_mte2 /
aic_fixpipe / aiv_vec / aiv_mte2 / aiv_mte3 / cube_utilization` 列，`aic_scalar_ratio` 列该 csv 未落盘，
取题面给定的 3.3%）：

| pipe | sglang 慢路径 @S=126 976 | sglang 快路径 @S=20 480 | vLLM consumer @S=126 976 |
|---|---|---|---|
| `aic_mac_ratio`（Cube MAC） | **0.016**（16.3 ms） | 0.209 | 0.277（13.7 ms） |
| `aic_mte1_ratio` | 0.014（14.2 ms） | — | 0.514（25.4 ms） |
| `aic_fixpipe_ratio` | 0.028（28.5 ms） | — | 0.359（17.8 ms） |
| `aiv_vec_ratio`（向量） | **0.065**（66.1 ms） | — | **0.915**（45.3 ms） |
| `aiv_mte2_ratio` | 0.013（13.2 ms） | — | 0.042（2.1 ms） |
| `aic_scalar_ratio` | 0.033（题面） | — | — |
| **busy 之和** | **0.181 ≈ 18%**（csv 6 列 0.148） | — | ≈ 2.3（多 pipe 并发，接近打满） |
| `cube_utilization(%)` | 98.99 | 92.6 | 97.58 |
| Duration | **1 016.6 ms** | 12.35 ms | **49.5 ms** |

三条判读：

1. **慢路径 ~82% 的周期没有任何 pipe 在干活**，唯一的长依赖链就是候选标量扫描（§2.3），
   且 §2.5 已用 15 cycle/op 把 985 ms 复算到 3% 以内 ⇒ 归因闭合。
   缺口只剩一项：`aiv_scalar_ratio`（预期 >80%）没落在 bundle 里，需上机补一次 profiler【未验证】。
2. **`cube_utilization` 在这里没有区分度**：慢 98.99%、快 92.6%、vLLM 97.58%，三者都在 92–99%。
   它度量的是「cube 参与占用」而不是「cube 在算 MAC」——真正的区分量是 `aic_mac_ratio`
   （0.016 vs 0.277，×17）和 `aiv_vec_ratio`（0.065 vs 0.915，×14）。
   ⇒ **不要在汇报里用 “cube_utilization=99% ⇒ cube 被打满/被打空” 来论证任何事**
   （题目给的 “cube_utilization=99% 说明 cube 空闲” 这一读法会被这个反例打穿：
   vLLM 同样是 97.58% 却是快路径）。
3. 快路径在 S=20 480 时 `aic_mac_ratio` 0.209 / `cube_utilization` 92.6 —— 一个正常的混合流水，
   与「快路径没有病」的判断一致。

---

## 3. 第 0 号问题：开源 def 无 candidate，镜像却在传 candidate

### 3.1 事实（不要再当矛盾）

| 侧 | 名字 | candidate 相关接口 |
|---|---|---|
| 镜像运行 | kernel `VllmAscendQuantLightningIndexerV2`（`MIX_AIC`, arch22） | 1 个算子吃 `candidate_mode∈{1,2,3}` + 输入 `candidate_topk_index` + 属性 `candidate_topk_blocks/blocks_size` + **输出** `candidate_out` |
| 开源 9.2.0 快照 | `QuantLightningIndexerV2` | **只有 source**：开关 `candidate_topk_blocks∈{-1,2048}`，**输出** `candidate_topk_index_out`，**无 candidate 输入** |
| 开源 9.2.0 快照 | `QuantSparseLightningIndexer`（同级 sibling） | **只有 consumer**：**输入** `candidate_topk_index`，无 `candidate_mode` |

【码】`experimental/attention/quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_def.cpp:78,94-97`；
`experimental/attention/quant_sparse_lightning_indexer/op_host/quant_sparse_lightning_indexer_def.cpp:76,99`；
`.../quant_lightning_indexer_v2/docs/aclnnQuantLightningIndexerV2.md:534-537`：
“第二级（consumer，候选块内选token级topk）**已拆分为独立算子** `aclnnQuantSparseLightningIndexer`：本接口不再有表达它的属性取值”。

### 3.2 结论：不是「快照更新/裁剪」，是**快照已经拆过、镜像还没拆**

- 【码】`quant_sparse_lightning_indexer/README.md:3`：“本算子由**拆分提交 P13** 从 `QuantLightningIndexerV2`
  的 `candidate_mode=2` 路径独立而来，因此不再注册 `candidate_mode` 属性”
- 【码】`.../quant_sparse_lightning_indexer_tiling.h:74,109`、`_tiling.cpp:392` 三处同口径注释。

⇒ 题面 §1.5-3 的三个猜测逐一判定：

| 猜测 | 判定 | 理由 |
|---|---|---|
| (a) 镜像 vendor 构建比 9.2.0 **新** | **否，反而更旧** | 镜像保留的是**拆分前**的三合一形态（有 `candidate_mode`）；9.2.0 已按 P13 拆完，`candidate_mode` 在 master 上不存在 |
| (b) DSV4.1 走的是另一个 aclnn | **否（名字层面）** | 调用链 `npu_quant_lightning_indexer_v3 → aclnnVllmAscendQuantLightningIndexerV2` 是**单个** vendor 算子，`VllmAscend` 前缀说明它是 vendor 在 `custom_transformer` 里自带的一份拷贝，不是 `quant_sparse_lightning_indexer` |
| (c) 快照被裁剪 | **否** | 功能没消失，搬到了 sibling `quant_sparse_lightning_indexer`（candidate 掩码实现就在那里，见 §4） |

⇒ **查证代理成立**：镜像那份「三合一」里的 `candidate_mode=2` 分支，与 9.2.0 的
`QuantSparseLightningIndexer` 是同一份代码的先后身。本文 §4 用后者做代理，并把「拆分时是否顺带改了算法」
标成【未验证】，给出上机一次性证伪的命令（§6）。

### 3.3 为什么镜像必须保留三合一

`vllm_ascend/models/deepseek_v41/indexer.py:274-288` 是**一次** `torch.ops._C_ascend.npu_quant_lightning_indexer_v3`
调用，用 `mode` 参数在同一个 kernel 里切换 source/consumer/现网【码】：

```python
mode = 1 if is_candidate_source else 2 if uses_candidate_filter else 3
selected, _, candidate_out = torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
    quantized_query, key, weights, query_scale, key_scale, topk, 2,
    block_table=..., metadata=..., candidate_topk_index=..., candidate_mode=mode, ...)
```

拆成两个 aclnn 后 Python 侧必须改成两次调用 + 显式候选张量，并处理「mode=3 不产候选」的分支；
vendor 显然选择了**保持单算子 API**。这与 `_cann_ops_custom/vendors/custom_transformer/` 的存在方式一致
（op_api / op_proto / op_impl 三件套由 vendor 自行构建，名字加 `VllmAscend` 前缀）【推】。

---

## 4. B) vLLM 版为什么在 S=122 880 下不掉进同一个坑

### 4.1 候选的表示：**按块号降序排好的「值+下标」对数组**，不是「按块号索引的掩码表」

【码】`experimental/attention/quant_sparse_lightning_indexer/op_kernel/arch22/quant_sparse_lightning_indexer_service_vector_arch22.h`：

```c
// :232  每行 2048 对 × 8B × 2 行 = 32KB，与 sglang 的 candBuf_ 同尺寸
pipe->InitBuffer(candBuf_, CeilDiv(s1BaseSize_,2) * BASE_TOPK * 2 * sizeof(float));
// :445  行分区：candPairs = candBuf[innerS1Idx * candBlocks * 2]   ← 按【候选秩】分区
// :446  if (info.isFirstS2InnerLoop)    ← 只在行首 tile 做一次
// :457-462  GM→UB DataCopyPad(candBlocks 个 int32) → Cast 成 fp32
// :464  DataCopy(candPairs[candBlocks], globalTopkIndice_, candBlocks)   ← 拼成 [value|idx] 对
// :467  SortAll(candPairs, candSortTmp, candBlocks)                      ← 按块号【降序】排序
```

关键点：**这块内存的大小 = `candBlocks`（恒 2048），与 S 完全无关**。
它是「候选集合」本身，不是「块号空间」的掩码。sglang 的 32 KB 装的是
`2 × CAND_TBL_CAP` 个**块号槽位**（`CAND_TBL_CAP=2752`，`:157`），vLLM 的 32 KB 装的是
`2 × 2048` 个**候选**（含块号+下标）。同样是 32 KB，一个随 S 增长、一个是常量。

### 4.2 每 tile 怎么查：两次二分 + 256 lane 距离最小值 + Brcb

⚠️ 先对齐一个**粒度差异**：该算子的 `s2BaseSize_ = 2048`【码】`op_host/quant_sparse_lightning_indexer_tiling.cpp:1532`
（`S2_BASE_SIZE = 2048`），是 sglang 的 512 的 **4 倍**。所以它的「tile」= 2048 个位置 = **256 个块**，
`numTiles = ceil(S/2048) = 60`（S=122 880），命中数 `h = 2048×256/15360 ≈ 34`。
下面的行号与常量都按这个粒度。

【码】同文件 `:435-505`，每 **(行, tile)**：

```c
// 1) 二分：候选里有几个 ≥ tileBlockBase / ≥ tileBlockBase+tileBlkNum  → [hi, lo) 就是「落在本 tile 的候选」
int32_t lo = CountGE(candPairs2, candBlocks, tileBlockBase);              // :476
int32_t hi = CountGE(candPairs2, candBlocks, tileBlockBase + tileBlkNum); // :477
//    CountGE :417-430 = 11 次标量 GetValue（降序序列上的 lower_bound）
// 2) 本 tile 的 256 个块号 vec = [base .. base+255]，acc 置 +inf
Cast(blkIdxF, globalTopkIndice_, ..., tileBlkNum);  Adds(blkIdxF, blkIdxF, tileBlockBase, tileBlkNum); // :481-483
Duplicate(acc.ReinterpretCast<int32_t>(), POS_INF_F32, tileBlkNum);                                   // :485
// 3) 对 [hi,lo) 内每个命中候选 v：acc = min(acc, |blkIdxF − v|)   ← 3 条 256-lane 向量指令 / 命中
for (j = hi; j < lo; j++) { Adds(diff, blkIdxF, -v, tileBlkNum); VecAbs(diff, diff, tileBlkNum); VecMinPair(acc, acc, diff, tileBlkNum); } // :488-494
// 4) 距离 0 即候选；clamp 到 1 得 isOut ∈ {0,1}；Brcb 展开到 2048 位置
Brcb(posDist, acc, tileBlkNum / 8, {1, 8});                 // :497    256 → 2048（32 repeat）
VecMinsScalar(posDist, posDist, 1.0f, s2BaseSize_);         // :499    dist<1 → 0，其余 → 1
candIsOut_ = posDist;                                       // :501
```

### 4.3 复杂度与内存（对照 §2；为可比，**按「每行扫完整个 S」归一**）

| 量 | sglang 快路径 | sglang legacy | vLLM consumer |
|---|---|---|---|
| 候选状态内存 | `2×CAND_TBL_CAP` floats = 22 KB，**容量 ∝ S** | 同左 | `2×2048` 对 = 32 KB，**∝ candBlocks，与 S 无关** |
| 每行一次性 | `Align(blkCnt,64)` SetValue + 2048 扫描 ≈ 1.9e4 标量 | — | 2048 DataCopy + Cast + `DataCopy` + **`SortAll(2048)`**（一次排序，非标量） |
| 每 (行,tile) 标量 | **0** | **4 676** | **22**（2×11 次二分探针） |
| 每 (行,tile) 向量 | **2 条**（64 + 512 lane） | 0 | **~8 条固定 + 3h 条**（256 lane）+ 1 条 2048 lane |
| tiles/行（S=122 880） | 240 | 240 | **60** |
| **每行向量 lane 总量** | 240×(64+512) ≈ **1.4e5** | 0 | 60×(8×256 + 34×3×256 + 2048) ≈ **1.7e6** |
| S 依赖 | `blkCnt` → 触发 CAP 回退 | 同左 | **无** |

【推】由 §4.1/§4.2 源码逐行计数 + §2.3 公式。
⇒ 归一之后：**vLLM 的掩码 lane 工作量是 sglang 快路径的 ~12 倍**（1.7e6 vs 1.4e5），
但它有 60 个 tile、每个 tile 的向量指令都是 256/2048 宽（流水满），而 sglang 只有 2 条窄指令。
**vLLM 的优势不是「每 tile 更省」，而是「内存不随 S 长」。**
（这条对 §5.5 很重要：vLLM 掩码本身并不便宜，它相对 source 调用贵 48.40/37.84 = 1.28× ≈ +10.6 ms。）

### 4.4 为什么 S=122 880 不触发回退：**根本没有可回退的东西**

- 【码】`grep -n "CAND_TBL_CAP\|_CAP\b\|Legacy" quant_sparse_lightning_indexer/op_kernel/arch22/*.h` → 无命中
  （sglang 侧有 `CAND_TBL_CAP :157`、`BuildCandidateMaskLegacyTile :511`、分派 `:576-588`）。
- 该算子唯一的容量常量是 `candBlocks ≤ BASE_TOPK = 2048`，由 **tiling 校验**保证
  （`quant_sparse_lightning_indexer_tiling.cpp:409-424`：candBlocks 由 `candidate_topk_index` 末维推导，
  且**当前只支持 2048**），与 S 无关。
- UB 预算 `144 + 32 + 8 = 184 KB ≤ 192 KB`【码】`:229-233`，与 sglang 的 `179.6 KB ≤ 192 KB`（`:200`）同量级
  ⇒ **两侧 UB 余量都只有 ~8–12 KB，都不允许「按块号稠密建表」**。这就是 sglang 必须设 CAP 的物理原因，
  也是 vLLM 一开始就不这么做（改成按候选存）的原因。

### 4.5 leak 降级语义：vLLM 也保留了，且做法与 sglang 等价

【码】同上文件 `:502-504`：
```c
// R11 (leak 语义, §1.6): isOut 仅用于分数降级 (pen 链)，索引保留真实位置号 —
// 候选外可达位置作为 topk 填充泄漏成有效索引 (对齐模型 where(idxs < compress_lens) 语义)，
// 不再把候选外索引改 -1 (原 isOutI32/CAST_RINT 链已删)
```
pen 链 `:653-657`：`pen = (NEG_HUGE − score) × isOut; score += pen` —— 只降分、不取消资格，
不可达位置仍是 `-inf` 沉底，索引只在不可达处为 -1。与 sglang 语义要求（
`sparse_lightning_indexer/README.md` 公式段 + 跨算子契约第 7 条：候选只降级排序、不取消入选资格；
输出 `-1` 槽仅出现在不可达位置）**逐条一致**。
差别只在降级值的构造：sglang 用位域技巧把全局位置序嵌进 `NEG_HUGE − p×2^76`
（`service_vector_arch22.h:718-736`，保证降级位按索引升序、跨 tile 归并无并列），vLLM 仍用裸 `−1e30`
（`candNegHuge_`）。**sglang 这条更强，不要为了对齐 vLLM 而丢掉。**

⇒ 结论：**语义层没有任何需要向 vLLM 借的东西**；两级语义（source 发布候选 → consumer 重打分）两侧同构
（vLLM：`candidate_source_layer_id` + `compress_ratios[src]==ratio` 决定哪一层发候选
【码】`/private/tmp/va0300/vllm_ascend/attention/dsa_v41.py:1240-1250`；
sglang：按 `(compress_ratio, compress_layer_id)` 分组发布、`_dsv41_index_qsli_topk` 消费
【码】`ascend_dsv4_backend.py:1708,1729-1755`）。

### 4.6 实测：全程无拐点

【数】`/private/tmp/va_calls.csv` 按 8 行/chunk 切分后，ratio-1 consumer 的 4 次中位：

| S | vLLM 4×consumer | ns/pair | sglang SLI | ns/pair |
|---|---|---|---|---|
| 4 096 | 3.98 ms | 0.2372 | 3.26 ms | 0.1945 |
| 8 192 | 9.09 ms | 0.2709 | 5.53 ms | 0.1649 |
| 16 384 | 20.40 ms | 0.3039 | 10.07 ms | 0.1501 |
| 20 480 | 24.76 ms | 0.2952 | 12.35 ms | 0.1472 |
| 24 576 | 25.32 ms | 0.2516 | **202.74 ms** | **2.0141** ← 拐点 |
| 65 536 | 33.97 ms | 0.1265 | 527.67 ms | 1.9657 |
| 122 880 | **48.40 ms** | **0.0962** | 984.66 ms | 1.9563 |
| 126 976 | 49.51 ms | 0.0952 | 1 016.66 ms | 1.9548 |

- vLLM 相邻 chunk 的最大跳变 = `24.76 → 25.32`（×1.023），**全程 1.02–1.09×，无拐点**。
- vLLM 在 **小 S 反而更差**（0.304 ns/pair @S=16 K），因为固定成本（每行一次 SortAll + 流水填充）摊不开；
  大 S 边际 0.056 ns/pair（24 576→126 976：`Δ23.6 ms / Δ102 400`）。
- sglang 在 S≤22 K 时 **比 vLLM 还快**（0.147 vs 0.295 @20 K）——再次说明算法本体没问题。
- 归一化后 S=122 880：`1.9563 / 0.0962 = 20.3×`；拆成 `13.2×（回退） × 1.54×（残差）`。

### 4.7 结论：sglang 该不该照抄 vLLM 的结构

**不该。** 一句可执行判断：

> 把 sglang 的块级表从 `float` 稠密改成 `1 bit/块` 的 bitmask（§5）。
> **不要**把候选改成「排序数组 + 每 tile 二分 + 距离最小值」——那会换掉数据结构却**增加**每 tile 工作量
> （按每行归一：~1.7e6 lane + 1.3e3 标量探针 vs 现在的 1.4e5 lane + 0 标量），
> 而残差里的 1.46× 在 topk/sort 管线，与掩码结构无关（§5.5）。

| 项 | 该不该移植 | 理由 |
|---|---|---|
| 候选→排序数组（O(candBlocks) 内存，无容量耦合） | **仅在未来需要 bs=1 或 S>256 K 时** | 代价是每行多 ~1.5e6 lane + 1.3e3 标量探针；当前 `bs∈[2,64] × S≤131 072` 用 bitmask 已全覆盖（§5.1） |
| 距离最小值（distance-min）公式，免「表→位置」展开 | 不必 | sglang 的 `Adds+Brcb`（2 条）已经比它（每行 ~1.7e6 lane）便宜一个数量级 |
| 「表按块号稠密」→「表按候选稀疏」这一**设计判断** | **值得写进注释/设计说明** | 这是 vLLM 唯一真正的结构优势：内存不随 context 长 |
| 两级 TopK 语义（source 产候选 / consumer 重打分） | **保留**，两侧已同构 | §4.5 |
| leak 降级（`NEG_HUGE−p×2^76` 位域构造，索引不缺位） | **必须保留 sglang 版本** | sglang 比 vLLM 的 `−1e30` 更强（全序、跨 tile 无并列）；vLLM 语义等价但更弱 |
| 相对块号、不加 `output_idx_offset` | **必须保留** | `sparse_lightning_indexer/README.md` 契约第 7 条；“The operator expands them itself, so the block ids pass through unchanged”【码】`ascend_dsv4_backend.py:1731-1737` |

---

## 5. C) 修复方案 + 预期

### 5.1 方案一（推荐）：块级 isOut 表 `float → 1 bit/块`

**UB 预算重算**（`candBuf_` 现为 `CAND_BUF_TOTAL=8192` floats = 32 KB，`:164`）：

| 区段 | 现状（float 表） | bitmask |
|---|---|---|
| 行 0 表 | `CAND_TBL_CAP=2752` floats = 11 008 B（S=122 880 实际需要 15 360 floats = 61 440 B，放不下） | `blkCnt` bits = 15 360/8 = **1 920 B** |
| 行 1 表 | 11 008 B | **1 920 B** |
| staging（Brcb 固定偏移） | 64 floats = 256 B | 256 B（不变） |
| 候选装载区 | 2048 int32 = 8 192 B | 8 192 B（不变） |
| bitMask 常量区（`1<<i`, i<32） | — | 32 int32 = 128 B（可放 `candConstBuf_`，`:203`） |
| tailPadding 守护 | 576 floats = 2 304 B | 0（不需要） |
| **小计** | **32 768 B（且只能装到 S=22 016）** | **≈12.4 KB（装到 S=131 072 仍余 ~20 KB）** |

覆盖域（`S ≤ 131 072` 是 DeepSeek-V4.1 的物理上限）：

| blockSize | blkCnt = ⌈131072/512⌉×(512/bs) | bitmask/行 | 2 行 + 装载 + staging | ≤ 32 KB？ |
|---|---|---|---|---|
| 8（主规格） | 16 384 | 2 048 B | 12.7 KB | ✅ |
| 2（最坏） | 65 536 | 8 192 B | 24.8 KB | ✅ |
| 16 / 32 / 64 | 8 192 / 4 096 / 2 048 | 1 024 / 512 / 256 B | ≤ 10 KB | ✅ |

⇒ **`bs∈{2,4,8,16,32,64}` 全值域 × `S≤131 072` 都不需要回退**，`CAND_TBL_CAP`（`:157`）与
`BuildCandidateMaskLegacyTile`（`:511-559`）可以整段删除（或降级为 `#if` 下的断言），
分派点 `:576-588` 变成无条件走快路径。

### 5.2 展开路径改造点（`ExpandCandMaskToPosition :381-438`）

题目担心的两点，逐个给出可执行答案：

**(1) 「Brcb 不能直接吃 bitmask」——对，但不能也不需要 vote/select。**
arch22 没有 `Scatter`（源码注释 `:379`「c220 亦未提供 Scatter」），且 `Brcb/vgather` 对
「随行分区变化的子张量」会读错数据（`:391-394`，实测结论）⇒ **staging 必须保留、Brcb 必须保留**。
只需要在 staging **之前**插一段纯向量「bit → 0.0f/1.0f」：

```c
// 每 (行, tile)：tileBlkNum=64 ⇒ 2 个 uint32 字，wordIdx = tileBlockBase / 32
LocalTensor<int32_t> bits = candBufUb_[CAND_BIT_OFFSET(innerS1Idx)];  // 行分区，1 bit/块
int32_t w0 = bits.GetValue(wordIdx);         // ← 仅剩的 2 次标量（每 tile 每行）
int32_t w1 = bits.GetValue(wordIdx + 1);
// a) 两字各广播成 32 lane → 64 lane 上得到 [1<<(i%32)] 的位掩码
Duplicate(word[0],  w0, 32);
Duplicate(word[32], w1, 32);
// b) 取位：And 掉位掩码 → 0 或 1<<i；Mins 1 → 0/1
And (word, word, candBitMaskVec);   // candBitMaskVec[i] = 1<<(i%32)，常量，一次预置（128 B）
Mins(word, word, oneVec);           // 0/1（1<<i ≥ 1 ⇒ 钳成 1）
// c) 0/1 → float 位型 0x00000000 / 0x3F800000（1×0x3F800000 = 0x3F800000，不溢出）
Muls(word, word, 0x3F800000);
// d) 搬到【固定偏移】staging（arch22 的 Brcb 只认固定偏移），再接原代码 :401/:419
Adds(stageF, word.ReinterpretCast<float>(), 0.0f, 64);
Brcb(candIsOutUb_, stageF, /*repeat*/8, {1, B32_VEC_REPEAT_STRIDE});
PipeBarrier<PIPE_V>();
```

指令账：**2 Duplicate + And + Mins + Muls + Adds + Brcb = 7 条向量**（≤64 lane 的 6 条 + 1 条 512 lane）
+ **2 次标量**，对比现状 2 条向量 / 0 标量 ⇒ 每 (行,tile) +5 条 64-lane 向量、+2 次标量。
整次调用增量 ≈ `5 × 4096 × 240 / 40 AIV = 1.2e5 条向量/AIV`（~1–2 c 各）
+ `2 × 4096 × 240 / 40 = 4.9e4 次标量/AIV`（~15 c，见 §2.5）
⇒ **+0.15 ms（向量）+ 0.41 ms（标量）≈ +0.56 ms**，即 75 ms 的 **0.7%**【推】。
（想去掉这 2 次标量：把 8 个字用一条 `Brcb` 广播成 8 lane×8 bit 的排布再 `And`，可做到 0 标量；
属可选优化，标记【未验证】—— 需确认该 Brcb 用法在 arch22 上不吃 `:391-394` 那个动态偏移坑。）

**(2) 表构建（`BuildCandBlockTable :444-506`）的改动**

```c
const int32_t wordLen = CeilDiv(blkCnt, 32);            // 15360/32 = 480
for (i = 0; i < wordLen; i++) tbl.SetValue(i, 0xFFFFFFFF);   // 全「非候选」，:458-460 换成 480 次而非 15 360 次
... DataCopyPad(candBlocks 个 int32) ...                     // :468 不变
// 命中置位：把 float-0 写改成 clear-bit（仍是纯 S pipe，:489-499 的 8 路展开照旧）
//   tbl[blk>>5] &= ~(1u << (blk&31))
```
标量成本：初始化从 `tblLen=15 360` 降到 **480**，命中置位从「写 float」变成「读-改-写 1 word」——
每行一次性，量级 ~1e4 → ~4e3，可忽略。

**(3) 语义等价性**（必须写进提交说明）：`1 = 非候选（isOut=1，降级）`、`0 = 候选`，
与现表 `1.0f/0.0f` 一一对应；`blkCnt` 之外仍按非候选处理（`:351-355` 的脏数据免疫不变：越界/-1 不落位），
`bs<8` 分支（`:427-437`，Brcb 32B 粒度不够的标量展开）在 bitmask 下同样可保留——只是从
「读表 1 次」变成「读 1 个 bit」。

### 5.3 为什么不选另两条路

| 备选 | 可行性 | 判定 |
|---|---|---|
| **uint8 表**（1 B/块，`Cast(u8→f32)` 一条指令展开） | S=131 072 × 2 行 = 32 KB，把 `candBuf_` 从 32 KB 撑到 41.2 KB，总 UB 188.8 KB（余量仅 3.2 KB）；且 `u8→f32` 未必有直接 Cast（需 u8→f16→f32 两条） | **不选**：省不了指令（同样 O(1) 向量），却吃掉全部 UB 余量，`bs=2` 形态直接放不下 |
| **照抄 vLLM 的排序数组** | 见 §4.3：掩码 lane 量 1.4e5 → 1.7e6（×12），另加每行 1.3e3 次标量二分探针 + 一次 `SortAll(2048)` | **不选**：解决的是「内存耦合」这个**我们已经在别处解决**的问题，净增成本 |
| **只把 legacy 循环做 8 路展开**（照抄 `CAND_SCAN_UNROLL`） | 可拿到 2–4×（`:480-500` 的同类收益），仍停在 `0.5–1.0 ns/pair` | **不选**：治不了根；且 `for` 里有 `if` 命中分支，展开收益远小于表构建 |

### 5.4 「改完 SLI ≈ 75 ms」的依据（三条独立，互相印证）

1. **快路径线性外推**【数】：快路径每 tile 时间恒定（S=4 K…20 K 的 ns/pair 0.194→0.147，主因是固定成本摊薄），
   `t(S) ≈ 0.3114 ms/tile × numTiles + 一次性`：
   `12.35 ms × (122 880/20 480) = 74.1 ms`（用 `bench_sli.py` 的 12.52 ms 则 75.1 ms）。
2. **自身 `aiv_vec_time` 硬下界**【数】：S=126 976 的慢调用里 `aiv_vec_ratio=0.065 × 1 016.6 ms = **66.1 ms**`
   ——这部分向量工作**已经存在**，掩码修好后它立刻变成关键路径。
   ⇒ 物理下界 **66 ms**，加上标量残余/AIC 交织，`70–80 ms` 是可信区间。
3. **同管线无候选算子对照**【数】：`LightningIndexerV2`（同 arch22、同 `SortAll+MergeSort`、
   同 s2Base=512，**不做候选掩码**）在 S=126 976 是 **81.49 ms**（0.1567 ns/pair）。

三条给出 **66 / 74 / 81 ms**，取 **≈75 ms**（保守口径 70–80）。
⇒ 与 §2.6 的 Amdahl 预测（74.7 + 0.3）一致；相对 985 ms 是 **×13.1**，
即 4 次/chunk 从 3.94 s 降到 ~0.30 s，占 chunk 算子时间从 74% 降到 ~5%。
整条 prefill 的 66.7 s → **约 5.4 s**（与 `INDEXER_T_S_FINDINGS.md §4` 的口径一致）。

### 5.5 残差分解：与 vLLM 0.096 ns/pair 的差距还剩多少、该不该追

修完后的同点对比（S=126 976，T=4096）：

| 量 | sglang SLI（修后 = 快路径） | vLLM consumer | 比 |
|---|---|---|---|
| 总时长 | **~75 ms**（预测；下界 66） | **49.51 ms** | **×1.52** |
| `aiv_vec_time`（向量） | **66.1 ms**【数】 | **45.3 ms**【数】 | **×1.46** |
| `aic_mac_time`（Cube MAC） | **16.3 ms**【数】 | **13.7 ms**【数】 | **×1.19** |
| `aic_mte1`（L1 装载） | 14.2 ms | 25.4 ms | ×0.56（sglang 更省） |
| `aic_fixpipe` | 28.5 ms | 17.8 ms | ×1.60 |
| `aiv_mte2`（GM→UB） | 13.2 ms | 2.1 ms | ×6.3 |
| ns/pair | ~0.149 | 0.0962 | ×1.55 |

**两个直接结论：**

1. **残差几乎全在向量侧**：`66.1 vs 45.3 = ×1.46`，而总时长比是 ×1.52 —— 向量一项就解释了残差的绝大部分。
   两个算子的打分/topk 管线是逐行同构的（都是 `Duplicate(−inf) → Adds → SortAll(cuS2LenVecAlign) →
   MergeSort(2048)`：sglang `:689-762` vs vLLM `:659-685`），所以这 1.46× 是**调优差异**
   （tile 切分、`AlignS2` 的排序对齐策略、INT8 让 `aiv_mte2` 少搬一半字节——sglang 13.2 ms vs vLLM 2.1 ms），
   不是算法差异。⇒ **要追就追这里**（单独一贴，别混进本次修复）。

2. **BF16→INT8 不值得**：
   - 实测同一逻辑调用的 Cube MAC 时间 **16.3 ms（BF16）vs 13.7 ms（INT8），只差 1.19×**。
     如果 INT8 真能拿到 2× cube 吞吐，这 16.3 应该掉到 8 ms —— 没有。
     【推】原因：这个算子的 cube 只做 `g=32 × head_dim=128` 的小 K 矩阵乘，`aic_mac_ratio` 慢路径下
     只占 1.6%、快路径 20.9%，**cube 不是瓶颈，换 dtype 也换不出时间**；
     且 INT8 要额外付 dequant/scale 链（vLLM 侧 `aic_fixpipe` 17.8 ms、`aic_mte1` 25.4 ms 都比 sglang 高）。
   - 换算到 75 ms 的预算里：`16.3 → 13.7` 省 **2.6 ms ≈ 3.5%**，而代价是重做 Q/K 的量化 prologue、
     精度风险、以及 q/k 缓存布局改造。
   - 【数】旁证：vLLM 为此在 indexer 桶里多养了 `_quantize_indexer_query_kernel` 384 次 **113.5 ms**
     和 `_prepare_indexer_indices_kernel` 384 次 **240.4 ms**（与本算子 1:1，合计 354 ms，占该桶 6.0%）
     ——sglang 侧 SLI 本身不吃这类成本（BF16 直入），但打包在 `index_prologue_qw_dsl` 里，未计入本贴。
   ⇒ **判定：不追。** 除非有别的收益（KV 显存、带宽），否则先修容量，再修向量常数。

### 5.6 落地清单（给算子团队，按优先级）

| # | 动作 | 位置 | 预期 |
|---|---|---|---|
| P0 | 块级表 → bitmask，删 `CAND_TBL_CAP`/legacy 分支 | `service_vector_arch22.h:157,511-559,576-588,458-460,489-499` | 985 → ~75 ms |
| P0 | 展开前插 bit→float 向量段（§5.2） | `:381-438`（bs==8 分支 `:396-402`） | +0.56 ms（向量 0.15 + 标量 0.41） |
| P1 | 补一条 `SLI_CAND_MASK_BITMASK=0` 回滚开关（照 `SLI_CAND_MASK_LEGACY` 的样式 `:361-363`） | 同上 | 可 A/B |
| P1 | UT：`bs∈{2,8,64} × S∈{4 K, 22 528（旧拐点）, 131 072}` × `candBlocks∈{64,2048}`，对 golden 比 `sparse_indices` | `tests/pytest/` | 覆盖旧回退域 |
| P2 | 向量侧常数（`AlignS2`、sort/merge 宽度） | 单独立项 | 最多再 ×1.46 |

### 5.7 第二处独立的优化：`virTopK` 按 `topk` 缩（比 bitmask 更小、可能更赚）

**问题**（【码】）：`virTopK = isSparseCountOver2K ? sparseCount : BASE_TOPK`（`:181`），
`sparseCount=512` 时仍取 `BASE_TOPK=2048` ⇒ 累加器/归并/抽取全部按 2048 宽维护，只导出 512。
**改动**（草案，3 处联动，缺一不可）：

| 位置 | 现在 | 改成 |
|---|---|---|
| `:181` | `virTopK = isSparseCountOver2K ? sparseCount : BASE_TOPK` | `virTopK = sparseCount`（或 `min(sparseCount, BASE_TOPK)` 兜底契约上限） |
| `:185` `sortOutBuf_` | `CeilDiv(s1BaseSize_,2) * virTopK * 2 * sizeof(float)` = 32 KB | 自动缩到 8 KB（topk=512） |
| `:760` `MergeSortVecCopy(..., virTopK, ...)` | dst = 2048 | dst = 512（**必须与累加器宽度一致**） |
| `:777-780` `offset/copyLen/copyNum` | 按 `SPARSE_COUNT_4K` 分档 | 复核：`offset=virTopK` 时 512 仍成立 |
| `:784` `Extract(outValueUb, outIdxUb, globalTopkUb_[...+2*i*offset], offset/32)` | `offset/32` 需是整数 | 512/32 = 16 ✓ |

**风险点**：`MergeSortVecCopy` 的语义是"512 个新 pair 与 dst 里已有的 pair 归并"，
dst 变窄后**必须保证旧累加器里的 512 个已经是最优的 512**（跨 tile 归纳），数学上成立（top-k 可结合），
但要复核 `InitSortOutBuf` 的填充宽度与 `virTopK` 一致（`:217`）。

**预期**：归并/抽取宽度 4× 下降；若这部分占 §5.5 里 66.1 ms 向量时间的一半，可望压到 ~45 ms 以内
（即**低于 vLLM 的 45.3 ms**）。**必须实测，不要写在验收口径里。**

**与 bitmask 的关系**：两者**正交**（一个管 UB 容量、一个管归并宽度），但都改同一个文件，
建议两个 commit、分别量收益，避免归因不清。

---

## 6. 验证清单（本机可做 / 必须上机）

**本机已做完（本贴全部数据）**
- [x] 形状级对账：256 行 vLLM / 144+144 行 sglang，per-chunk 8 次、3/1/4 拓扑（§1.2–1.3）
- [x] 复杂度公式代入 → 0.958 s vs 实测 0.985 s（§2.5）
- [x] 计数核对（§2.7）
- [x] 候选结构代理：9.2.0 `quant_sparse_lightning_indexer` 的 BuildCandidateMask（§4）

**必须上机（本机无法闭环，按重要性排序）**
1. 【未验证】`aiv_scalar_ratio`（慢路径预期 >80%）。一条 `profiler` 即可，bundle 里这一列被裁掉了。
2. 【未验证】镜像那份 `VllmAscendQuantLightningIndexerV2` 的 `candidate_mode=2` 分支是否真用
   「排序数组 + CountGE」结构（§4 用的是 9.2.0 拆分后的同源代码做代理）。一次性证伪命令：
   ```bash
   # 在镜像内
   strings .../op_api/lib/libcust_opapi.so | grep -iE "candidate|mode" | head -40
   ls  .../op_impl/ai_core/tbe/custom_transformer_impl/dynamic/ | grep -i lightning
   # 打开 fused_quant_lightning_indexer_manage.py，看 candidate 分支里是否有
   #   CountGE / SortAll(candPairs) / Brcb(posDist)  —— 有 ⇒ 与 §4 同构
   ```
   （镜像内路径来自 `_tmp_preflight/out/node-61-47-19-71_20261126_145650/21_py_only_in_image.txt`
   与 `40_custom_ops.txt`；**该 vendor 树没有进 git**，`vllm_ascend/_cann_ops_custom/` 只有 `.gitkeep`【数】。）
3. 【未验证】`Mins(int32)` / `Muls(int32, 0x3F800000)` 在 arch22 上是否按预期出 0/1 → 0.0f/1.0f
   （若 `Mins` 不支持 int32，退化为 `Duplicate(oneVec)` + `Min`，指令数 +1）。
4. 【未验证】bitmask 版在 `bs=2, S=131 072` 的 UB 实测占用（纸面 24.8 KB，需看编译器对齐后是否仍 ≤32 KB）。
5. 【未验证】修后实测：`bench_sli.py` 扫 S=4 K…131 K，确认无拐点且 S=122 880 落在 70–80 ms。

---

## 7. 溯源

### 7.1 sglang / ops-transformer（只读）

| 用途 | 路径:行 |
|---|---|
| 快路径展开 | `experimental/attention/sparse_lightning_indexer/op_kernel/arch22/sparse_lightning_indexer_service_vector_arch22.h:381-438` |
| 块级表构建 | 同上 `:444-506`（`:450` tblLen、`:468` 装载、`:480-500` 8 路展开） |
| legacy 逐 tile 扫描 | 同上 `:511-559`（`:544-546` 512 次 init、`:547-555` 2048 次 GetValue） |
| 分派 / 阈值 | 同上 `:562-589`；`CAND_TBL_CAP=2752 :157`；`candBuf_` 布局 `:148-170`；`:200` UB 预算 179.6 KB |
| 调用点（每行每 tile） | 同上 `:641`（行循环）、`:707`（`BuildCandidateMask`） |
| s2 tile 循环 | `op_kernel/arch22/sparse_lightning_indexer_kernel_arch22.h:395,723` |
| tiling 常量 | `op_host/sparse_lightning_indexer_tiling.cpp:1098(S2=512),1104(S1=8),1027-1029` |
| candidate 契约 | `sparse_lightning_indexer/README.md`（公式段 + 约束 7 条）；`op_host/*_def.cpp:81` |
| Brcb/vgather 限制、无 Scatter | `..._service_vector_arch22.h:375-379,391-394` |
| vLLM 侧代理实现 | `experimental/attention/quant_sparse_lightning_indexer/op_kernel/arch22/quant_sparse_lightning_indexer_service_vector_arch22.h:232,415-430,435-505,640-685` |
| 拆分证据 | `quant_sparse_lightning_indexer/README.md:3`；`_tiling.h:74,109`；`_tiling.cpp:392`；`quant_lightning_indexer_v2/docs/aclnnQuantLightningIndexerV2.md:534-537`；`quant_lightning_indexer_v2/op_host/*_def.cpp:78,94-97`（无 candidate 输入） |
| sglang 发起处 | `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py:1729,1747,1755` |

### 7.2 vLLM-ascend（只读，worktree `/private/tmp/va0300`）

| 用途 | 路径:行 |
|---|---|
| mode / 单算子三合一 | `vllm_ascend/models/deepseek_v41/indexer.py:274-288` |
| 候选形状 | 同上 `:191-206`（`candidate_shape = (tokens,1,candidate_topk_blocks)`） |
| 候选来源层 | `vllm_ascend/attention/dsa_v41.py:1240-1250,1271-1272` |
| vendor 树不在 git | `git ls-tree -r origin/dsv41-perf -- vllm_ascend/_cann_ops_custom` → 只有 `.gitkeep` |

### 7.3 数据

| 用途 | 路径 |
|---|---|
| sglang 逐次调用（288 行） | `/Users/an/Developer/sglang/_tmp_preflight/profile_bundle/indexer_calls_cold.csv` |
| sglang pipe 计数 | `.../profile_bundle/prof_128k_cold/kernel_top.csv` |
| vLLM 逐次调用（256 行） | **`/private/tmp/va_calls.csv`**（题面写的 `profile_bundle/k128k/va_calls.csv` 不存在，§1.4） |
| vLLM 桶/形状/计数 | `/private/tmp/va0300/_tmp_preflight/profile_bundle/k128k/{va_buckets.csv,kernel_top.csv,va_summary.md,meta.txt}` |
| 微基准拐点表 | `_tmp_preflight/bench_sli.py`（结果抄自 `INDEXER_CAP_FINDING.md`） |

### 7.4 本贴修正的上游口径（§1.2–1.4）

1. vLLM 每 chunk 索引器算子调用 = **12 次**（`op_statistic_top.csv` Count=384 = 32×12）；
   `va_calls.csv` 只保留 `Duration>0.5 ms` 的 8 次（= 3 半维 + 1 leading + 4 consumer），对账用这 8 次。
2. 两侧层拓扑相同（5 个 ratio-1 + 3 个 ratio-2），**不存在 4/4 vs 5/3 的差异**；差异只在
   ratio-2 是否走两级（vLLM 走、sglang 不走），且不在主矛盾上。
3. `va_calls.csv` 实际在 `/private/tmp/va_calls.csv`；`dsv41-perf` 的 worktree 是 `/private/tmp/va0300`
   （`/Users/an/Developer/vllm-ascend` 停在 `main`，其 `vllm_ascend/_cann_ops_custom/` 只有 `.gitkeep`）。
4. `cube_utilization`（98.99 / 97.58）**不能**用来论证「cube 空闲」；有区分度的是
   `aic_mac_ratio`（0.016 / 0.277）与 `aiv_vec_ratio`（0.065 / 0.915）。
