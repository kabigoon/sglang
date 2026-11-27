# sglang 索引器 t(S) 实测（cold + EP 两份 trace）—— 给 vLLM-ascend 对比用

数据：`profile_bundle/indexer_calls_cold.csv` / `indexer_calls_ep.csv`
（由 `dump_indexer_calls.py` 从 `prof_128k_cold_1124_124110` / `prof_128k_deepep_1124_132655`
的 `kernel_details.csv` 导出；S = block_table 宽 × page_size(64)，T = q 的第 0 维。）

## 1. 模型：成本 ∝ T_q × S（不是 ∝ S）

| 量 | cold | EP |
|---|---|---|
| `SparseLightningIndexer` 调用数 / 总时间 | 128 / **66.7 s** | 128 / 66.6 s |
| `LightningIndexerV2` 调用数 / 总时间 | 128 / 2.8 s | — |

**验证**：最后一个 chunk 是**残 chunk**（T_q = 130 176 − 126 976 = 3 200，不是 4 096）。
它的 SLI 实测 787.2 ms，而前一档（S=126 976，T_q=4096）是 1016.7 ms：

```
787.2 / 1016.7 = 0.774      vs      3200 / 4096 = 0.781      （差 0.9%）
```

⇒ **归一到 `ns per (q·k) pair` 之后才可比**，否则残 chunk 会污染曲线。

## 2. 两段 + 一个 16× 拐点（两份 trace 完全一致）

`SparseLightningIndexer`（S ≥ 4 096，每档 4 次调用取中位）：

| 区间 | 拟合 | 归一化 |
|---|---|---|
| **S ≤ 20 480**（20 次调用） | `t ≈ 0.99 ms + 0.554 µs/row × S` | **0.155 ns/(q·k)** |
| **S ≥ 24 576**（108 次调用） | `t ≈ 36.1 ms + 7.456 µs/row × S` | **1.964 ns/(q·k)** |
| 整体单线（会得到负截距） | `t ≈ −30.0 ms + 8.157 µs/row × S` | — |

- **拐点位置：S ∈ (20 480, 24 576]** —— `S=20 480 → 12.3 ms`，`S=24 576 → 202.7 ms`，**×16.4**。
- cold 与 EP 的拐点、斜率**逐点重合**（8.157 vs 8.103 µs/row，差 0.7%）⇒ **部署无关、实现固有**。
- **时间分布极度偏斜**：快段 20 次调用只花 **156 ms（0.2%）**，慢段 108 次花 **66.5 s（99.8%）**。

对照 `LightningIndexerV2`：**线性、无拐点**，`t ≈ 0.66 µs/row`（≈0.16 ns/(q·k)），128 次共 2.8 s。

## 3. 对 vLLM 对比的口径（重要）

1. **同 S 对比请用 LIV2**：`S=122 880` 时 LIV2 = **78.9 ms**（cold）/ 81.5 ms（demo trace），
   而 SLI = **984.5 ms**。后者的 900 ms 差距来自拐点，不是"索引器"这个算法本身。
2. **曲线形状比斜率更有信息量**：sglang 的索引器**在小 S 区间（0.155 ns/pair）比自家 LIV2
   （0.16 ns/pair）还快**，说明算法没问题；**是超过约 20 K 后掉进一条慢 12.7× 的路径**。
   如果 vLLM 是全程线性，那么结论是"sglang 实现引入了一个容量拐点"，而非"算子必然慢"。
3. **别看总时间不看 S**：50% 前缀命中的生产配置下所有 chunk 从 S≈68 K 起步 ⇒ 100% 落在慢段，
   这就是 TTFT 73.7 s 里 67 s 的来源。

## 4. 给算子团队的一句话

> 修掉拐点（让慢段回到快段的 0.155 ns/pair），SLI 在 128K prefill 的合计时间会从
> **66.7 s 降到约 5.4 s**（66.5 s × 0.155/1.964 + 0.16 s），TTFT 随之从 73.7 s 进到 ~13 s 量级。

## 5. 未验证 / 待办

- 拐点的**物理原因**未定：S=16 384 = `candidate_topk_blocks(2048) × candidate_block_size(8)`，
  是"候选覆盖饱和"的天然阈值，但实测拐点在 20 480 之后 —— 需要算子侧确认是否是
  UB/workspace/LD-topk 结构（`TOPK_MAX_SIZE=8192`）的容量上限触发 fallback。
- 只在 A3 + CANN 9.2.0 + `sgl_kernel_npu 2026.11.0.post4` 上测过。
