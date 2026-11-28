# vLLM 侧实测数据副本（原路径在 `/private/tmp`，换机器必备）

这些文件**原本不在任何 git 仓库里**：它们是上一台机器在 vLLM-ascend 的
`dsv41-perf` worktree（`/private/tmp/va0300`）里跑 profiler 得到的产物。
为了让交接分支自包含，把**必要的小文件**复制到这里。

## 文件说明

| 文件 | 原路径 | 行数 | 用途 |
|---|---|---|---|
| `va_calls.csv` | `/private/tmp/va_calls.csv` | 6 704（indexer **256** 行） | 逐次调用明细；**按 8 行一组 = 1 个 chunk**（3×page64 + 1×leading + 4×page128） |
| `op_statistic_top.csv` | `/private/tmp/va0300/_tmp_preflight/profile_bundle/k128k/` | 6 | `VllmAscendQuantLightningIndexerV2` **Count=384**（32 prefill×8 + 16 decode×8）+ 两个辅助 kernel 各 384 次 |
| `kernel_top.csv` | 同上 | 176 行 VllmAscend | pipe 计数（`aic_mac_ratio=0.277` / `aiv_vec_ratio=0.915`） |
| `va_buckets.csv` | 同上 | — | 算子桶（indexer 桶 1248 kernels / 5 892.6 ms） |
| `va_summary.md` | 同上 | — | 65 步、切步口径、来源 profiler 路径 |
| `meta.txt` | 同上 | — | 原始 `kernel_details.csv` 的 46 列定义（**原始 49 MB 文件未复制**） |
| `va_steps.csv` / `step_trace_time.csv` | 同上 | — | 逐 step 时间线 |
| `out/node-61-47-19-71_*/` | `/private/tmp/va0300/_tmp_preflight/out/` | 15 文件 | 镜像 vendor 树的**文件清单**（`40_custom_ops.txt`、`21_py_only_in_image.txt`）、flags、版本、dsa_v41 fork-vs-image diff |
| `out/fail_.../12_va_probe.txt` | 同上 | — | **480 行探针**：`index_source` / `candidate_source` / `uses_cfilter` / `cand=2048x8` 组合分布 |

## 三个使用注意

1. **`va_calls.csv` 只保留 `Duration > 0.5 ms` 的调用**（384 → 256）。缺的 128 行是 decode / 廉价调用。
2. **`Input Shapes` 里的 S 是预分配容量**（恒定 `54 347`），**不是实时 S**。
   实时 S 必须按调用时序重建：第 j 个 chunk 的 `S ≈ (j+1) × 4096`。
3. vendor 自定义算子的**源码内容没有抓到**（只有清单）。要看实现必须上机（见 `HANDOFF_README.md` §7-2）。
