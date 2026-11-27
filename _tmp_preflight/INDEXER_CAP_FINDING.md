# 结论：`SparseLightningIndexer`（BF16, arch22）慢在"容量阈值后的标量回退"

## 一句话

kernel 里候选掩码有两条实现：**P1 快路径**（块级表一次建 + Brcb 展开，O(blkCnt+tiles)）和
**legacy 标量路径**（每 512 位置的 tile 重扫 2048 个候选块，O(tiles×candBlocks)）。
当 `blkCnt > CAND_TBL_CAP(2752)` —— 即 **S > 22 016**（`ceil(S/512)×(512/8)`）—— 会**静默**切到后者，慢 **13×**。
128K/50% 前缀命中的生产场景每个 chunk 的 S 都 ≥68K ⇒ **100% 落在慢路径**。

## 机制（源码坐标）

```c
// ops-transformer/experimental/attention/sparse_lightning_indexer/op_kernel/arch22/
//   sparse_lightning_indexer_service_vector_arch22.h
constexpr static uint32_t CAND_TBL_CAP = 2752;                       // :157 块级 isOut 表容量（每行分区）
const int32_t tileBlkNum  = s2BaseSize_ / blockSize;                 // :568 512/8 = 64
const int32_t blkCnt      = CeilDiv(actS2Size, 512) * tileBlkNum;    // :572 = ceil(S/512)*64
if (blkCnt <= CAND_TBL_CAP) { …快路径（:574-582）… }                  // :573
else                        { BuildCandidateMaskLegacyTile(...); }    // :584 → :508 标量逐 tile 重扫
```

`blockSize = candidate_block_size = 8` ⇒ 阈值 `ceil(S/512) ≤ 43` ⇒ **S ≤ 22 016 走快路径**。

| S | tiles | blkCnt | 路径 | 实测 |
|---|---|---|---|---|
| 20 480 | 40 | 2 560 | 快 | 12.52 ms |
| 21 504 | 42 | 2 688 | 快 | 13.02 ms |
| **22 016** | **43** | **2 752（=CAP）** | **快（边界）** | **13.39 ms** |
| **22 528** | **44** | **2 816（>CAP）** | **回退** | **174.80 ms（×13.1）** |
| 122 880 | 240 | 15 360（5.6×CAP） | 回退 | **985.44 ms** |

## 证据（三条独立测量）

1. 真实 trace（`prof_128k_cold` / `prof_128k_deepep`，128K 冷启动，144 次调用）：S=122 880 处 **984.5 / 984.7 ms**；
   完整 S 扫描呈两段：`S≤20 480: 0.554 µs/row` vs `S≥24 576: 7.456 µs/row`（12.7×），拐点在 (20 480, 24 576]。
2. 独立微基准（`bench_sli.py`，脱离模型、连续页池）：S=122 880 → **985.44 ms**（与 trace 差 0.1%）；
   并把拐点夹到 **22 016（快）/ 22 528（慢）**，与 `CAND_TBL_CAP` 推导逐字吻合。
3. 归一化：快路径 **0.1485 ns/pair**（S=22 016）≈ LIV2 的 0.16 ns/pair ⇒ **算法本身没问题，慢的是回退**。

## 影响面

- 只影响 `S > 22 016` 的调用 ⇒ 128K 场景全中；短上下文（S<22K）不受影响，所以在短请求上"看不出问题"。
- 每 chunk（50% 命中，S=68K~128K）：4 次 SLI × ~0.5–1.0 s = **约 4 s**，占该 chunk 算子时间 5.4 s 的 **74%**；
  整条 prefill 的 66.7 s / TTFT 73.7 s 里约 **67 s** 来自它。
- 与 vLLM-ascend 的 20.4× 差距可分解为：**12.7×（回退）× 1.6×（BF16 vs INT8 / kernel 质量）**——主因是前者。

## 修法（给算子团队）

块级 isOut 表当前是 **float（4 B/块）**：S=131 072 需要 16 384 块/行 ⇒ **64 KB/行**，
而 `candBuf_` 总共只有 32 KB（8192 floats，其中 2×2752=5504 floats 就是这张表）⇒ 放不下，这就是 CAP=2752 的由来。

> **改成 bitmask（1 bit/块）**：S=131 072 只需 **2 KB/行**（2 行共 4 KB），余量 30×。
> 展开改位运算后，快路径可覆盖整个 128K；预期 **SLI ≈ 75 ms**（与无候选的 LIV2 持平）。

## 汇报时不要说的三句话（会被追问穿）

| 别这么说 | 应该说 |
|---|---|
| "prefix token 多了就慢" | "**S 超过 22 016 这个硬阈值**就换实现；50% 缓存下每个 chunk 都超过" |
| "这个算子本身慢 / BF16 慢" | "**同一个 kernel 的快路径比自家 LIV2 还快**；慢的是容量不够触发的回退" |
| "升级 ops-transformer 就好了" | "**P1 重写已经装了**（快路径就是它）；它的容量只到 S=22K，需要扩容量" |
