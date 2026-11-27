# vLLM-Ascend 对照实验全记录（2026-11-26）

> 目的：在 A3×16 上用 **DeepSeek-V4.1-Flash(w4a8)**、同一口径（128K 输入 / C=1 / 4096 chunk）跑 vLLM-Ascend，
> 拿到 TTFT 与 sglang 基线（**73.7 s**）对比，并**用 profiler 定位差距落在哪个算子上**。
>
> 结论：**vLLM 27.4 s vs sglang 73.7 s（2.7×）**；差距几乎全部来自**索引器（每 chunk 4.00 s → 0.172 s，23×）**，
> MLA 完全一致、mHC 快 3×、通信反而是 vLLM 更重。
> 本文档不是 PR 的一部分，仅作为 `_tmp_preflight/` 的调研记录保留。

---

## 1. 三台机器、三种命运

| 机器 | HDK | 结果 |
|---|---|---|
| node-61-47-19-71 | **25.5.1** | ❌ 卡在 2 个 vector-core 崩溃 + 1 个自定义算子未注册（见 §3） |
| localhost（另一台） | **25.5.0** | ❌ 卡在 `aclnnHcPre not in libopapi.so` |
| **node-61-47-19-68** | **26.1.1** | ✅ **开箱可用**：纯镜像 + 官方默认参数，零 workaround |

镜像统一为 `quay.io/ascend/vllm-ascend:v0.30.0-a3`（`77aea9096250`，2026-10-29）：
CANN **9.2.1** / torch **2.10.0+cpu** / torch_npu **2.10.0.post7** / Python 3.12.16，vllm 与源码树均 editable 装在 `/vllm-workspace/vllm-ascend`。

vllm-ascend v0.30.0 的官方验证组合是 **CANN 9.2.1 ↔ HDK 26.2.RC1**。三台机器的差别只有 HDK，而结果从"完全跑不起来"到"开箱可用" —— 说明 **HDK 版本是这条链路上最硬的门槛**。

---

## 2. HDK 26.1.1 上的可用配置（本次对比用的口径）

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export VLLM_LOG_STATS_INTERVAL=2
export VLLM_TORCH_PROFILER_DIR=/home/a00821909/va_prof      # 仅在要采 profile 时

vllm serve /home/weights/DeepSeek-V4.1-Flash-1017-w4a8 \
  --host 0.0.0.0 --port 8900 --served-model-name dsv41 \
  --max-model-len 150000 --max-num-batched-tokens 4096 --max-num-seqs 1 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 4 --tensor-parallel-size 4 --enable-expert-parallel \
  --quantization ascend --block-size 128 \
  --safetensors-load-strategy lazy --tokenizer-mode deepseek_v4 \
  --enable-prefix-caching --reasoning-parser deepseek_r1 \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":true}'
```

与 sglang 侧同口径的三点：**4096 chunk / C=1 / 不开投机**；图按用户环境默认开启（sglang 基线也是开图的）。
注意 `--tokenizer-mode` 在 v0.30.0 只认 `deepseek_v4`（没有 `deepseek_v41`）。

---

## 3. 在 HDK 25.5.x 上踩到的 5 个坑（对后来者有用）

| # | 症状 | 根因 | 状态 |
|---|---|---|---|
| 1 | `fault kernel_name=aiv_all_gather_int32_t`，`ErrCode=507035 vector core exception`，`tag[AllGather_group_name_123]`、`count[384]`（= 384 个专家的计数），卡 43 s 后炸 | 官方 A3 示例设 `HCCL_OP_EXPANSION_MODE=AIV`（小集合通信展开到 vector core），该 kernel 在 25.5.x 不可用 | ✅ `unset` 后 `aiv_*` 归零 |
| 2 | `fault kernel_name=_gather_lookback_kernel`，同类 vector core exception | **Engram 的 Triton kernel**（`ops/triton/engram_lookback.py`，被 `engram/model_state.py:117` 调用），triton-ascend 3.6 生成码与 25.5.x 不兼容 | 🟡 有 torch 等价实现兜底 |
| 3 | 容器内 `devices: 0` / `libascend_hal.so` 找不到 | 宿主没有 Ascend Docker Runtime | ✅ 手动 `--device=/dev/davinci0..15` + `davinci_manager/devmm_svm/hisi_hdc` + 挂 driver/dcmi/npu-smi；`ASCEND_RT_VISIBLE_DEVICES=0-15` 的区间写法**不认**，会只看到 1 卡 |
| 4 | 容器内 `import numpy` 崩（`numpy/ma/core.py` 缺失）、`pip` 也崩（`pip._vendor.idna.core` 缺失），`find site-packages -name core.py \| wc -l` = **0** | **-71 那台的镜像副本被按 `core.py` 批量删过**（疑似他人的清理脚本）；同一镜像在另一台 `core.py=36`、`import numpy/torch/torch_npu` 全 OK | ✅ 换机器（镜像本身没坏） |
| 5 | `aclnnHcPre or aclnnHcPreGetWorkspaceSize not in libopapi.so` | 自定义算子包未被 ACLNN 看到（`ASCEND_CUSTOM_OPP_PATH` 未指向 `_cann_ops_custom/vendors`） | ❌ 未解决（在 26.1.1 上不复现） |

**经验**：新镜像 + 老驱动，最先炸的不是 CANN 初始化，而是**新的执行路径**（AIV 展开的集合通信、triton-ascend 生成的 kernel）。
小算子冒烟（bf16 matmul、2 卡 HCCL all_reduce）能过，**不代表真实 forward 能过**。

---

## 4. 主结果：TTFT 对比（128K 输入，C=1，4096 chunk）

| 指标 | sglang 基线 | **vLLM-Ascend** | 备注 |
|---|---|---|---|
| **TTFT** | **73.7 s** | **27.4 s**（profiled）/ 26.6–31.0 s（无 profiler） | 2 条请求时 26.6 + 35.4；单条冷启动 26.6–27.4 |
| TPOT | — | **11.55–13.9 ms** | decode 正常 |
| 总吞吐 | — | 4,088–4,631 tok/s | 串行口径 |
| 每 4096-chunk 墙钟 | **4.6 s** | **0.83–0.86 s** | **≈5.4×** |

> sglang 的 73.7 s 对应"128K 请求、50% 前缀命中、实算 64K（16 chunk）"；
> vLLM 的 26.6–27.4 s 对应"128K 全量实算（32 chunk）"。
> 按**每 chunk 成本**比较才是同口径：**4.6 s vs 0.85 s**。

一致性检查：我们把 profiler 常开的那次 TTFT（27.4 s）与不带 profiler 的冷启动（26.6 s）相比，差 3% —— **这条链路上的 profiler 开销可忽略**（与 sglang 侧 ~300us/kernel 的膨胀不同）。

---

## 5. profiling 怎么采（附两个坑）

**触发方式**：vLLM 的 `/start_profile`、`/stop_profile` 路由**只在启动时给了 `--profiler-config` 才注册**（`entrypoints/serve/profile/api_router.py::attach_router` 检查 `app.state.args.profiler_config.profiler`）。
只设 `VLLM_TORCH_PROFILER_DIR` 会出现 `POST /start_profile → 404`。

```bash
vllm serve ... \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/home/a00821909/va_prof"}'
# 触发
curl -s -X POST http://127.0.0.1:8900/start_profile
curl -s -X POST http://127.0.0.1:8900/stop_profile
```

**坑 1：worker 是 daemon 进程，采集完不会自动解析成 CSV**

```
[ERROR] profiler.py: The profiling data cannot be parsed during the daemon process,
it is recommended that you use an offline parsing interface ...
from torch_npu.profiler.profiler import analyse
analyse("profiling_data_path")
```

⇒ 采集只产出原始 `PROF_*`（每个 rank ~540–800 MB）。必须**另起一个非 daemon 进程**离线解析：

```bash
/usr/local/python3.12.16/bin/python3 -c \
  "from torch_npu.profiler.profiler import analyse; analyse('<..._ascend_pt>')"
```
（一个 800 MB rank 约需 **3 分钟**；解析后得到 `ASCEND_PROFILER_OUTPUT/{kernel_details,op_statistic,step_trace_time,...}.csv`）

**坑 2：容器里的 `msprof` 用不了**
`ascend-toolkit/latest` 指向镜像自带 9.2.1，但该版本 `msprof` **没有 `--input`**（位置参数被当成 `--application`），
**不支持离线导出**；宿主那台只有 8.5.0 的 `msprof`，解析不了 9.2.1 数据。⇒ 只能走上面的 `analyse()`。

**DP4 的取舍**：一次采集会写 **16 份** rank 数据（~8.7 GB）。
**拿到请求的那个 DP 组数据明显更大**（本次 `dp2_*` 是 797 MB，其它组 540 MB）—— 只解析 `dp2` 的 rank0 即可。

---

## 6. 时间线拆解（`step_trace_time.csv`）

```
Computing                    21.166 s   77.8%
Communication (not overlapped) 4.706 s  17.3%
Overlapped (comm)             0.028 s    0.1%
Free                          1.320 s    4.9%
──────────────────────────────────────────────
Stage total                  27.191 s   ← 与 profiled TTFT 27.4 s 吻合
```

算子口径（`kernel_details.csv`）总 kernel 时间 **29.983 s / 146,304 个 kernel**，
比时间线多出的 ~2.8 s 是**通信与计算重叠**的部分。

---

## 7. 算子桶（修正驼峰命名后）

| 桶 | kernel 数 | 总耗时 (ms) | 占比 | 平均 (us) | top 算子 |
|---|---|---|---|---|---|
| **comm** | 22,104 | **8,208.3** | **27.38%** | 371 | alltoallv 1784 / allreduce 1061 / allgather 619 |
| **indexer** | 1,248 | **5,892.6** | **19.65%** | 4,722 | `VllmAscendQuantLightningIndexerV2` 5518 |
| **mla_attn** | 2,064 | **4,885.4** | **16.29%** | 2,367 | `SparseFlashMla` 4788 |
| **mhc** | 7,680 | **3,909.4** | **13.04%** | 509 | `HcPre` 2510 + `HcPost` 1399 |
| gemm | 10,743 | 2,157.8 | 7.20% | 201 | MatMulV3 1096 / MatMulV2 713 / GroupedMatmul 348 |
| other | 42,203 | 1,492.5 | 4.98% | 35 | ScatterNdUpdateSk 385 / RmsNorm 227 |
| norm | 36,184 | 1,158.1 | 3.86% | 32 | TransposeBatchMatMul 420 / RmsNormCast 225 |
| moe | 12,160 | 1,142.0 | 3.81% | 94 | GroupedMatmulSwigluQuantV2 616 / MoeTokenUnpermute 139 |
| quant | 11,918 | 1,136.6 | 3.79% | 95 | QuantBatchMatmulV3 798 / DynamicQuant 269 |

---

## 8. 关键算子明细（`op_statistic.csv`；384 = 32 chunk × 12 层）

| 算子 | Core | 次数 | 总耗时 | 平均 | 最大 | 占比 |
|---|---|---|---|---|---|---|
| `VllmAscendQuantLightningIndexerV2` | MIX_AIC | 384 | **5.518 s** | 14.37 ms | 49.52 ms | 21.86% |
| `SparseFlashMla` | MIX_AIC | 1,920 | 4.788 s | 2.494 ms | 4.536 ms | 18.96% |
| `HcPre` | MIX_AIC | 3,840 | **2.510 s** | 0.654 ms | 1.024 ms | 9.94% |
| `alltoallvAicpuKernel` | AI_CPU | 3,840 | 1.784 s | 0.465 ms | 2.883 ms | 7.07% |
| `HcPost` | AI_VECTOR_CORE | 3,840 | **1.399 s** | 0.364 ms | 1.079 ms | 5.54% |
| `MatMulV3` | AI_CORE | 2,542 | 1.096 s | 0.431 ms | 5.773 ms | 4.34% |
| `allreduceAicpuKernel` | AI_CPU | 3,895 | 1.061 s | 0.273 ms | 1.925 ms | 4.20% |
| `MatMulV2` | AI_CORE | 6,281 | 0.713 s | 0.114 ms | 0.599 ms | 2.83% |
| `QuantBatchMatmulV3` (MIX) | MIX_AIC | 4,224 | 0.643 s | 0.152 ms | 0.329 ms | 2.55% |
| `allgatherAicpuKernel` | AI_CPU | 3,277 | 0.619 s | 0.189 ms | 1.611 ms | 2.45% |
| `GroupedMatmulSwigluQuantV2` | MIX_AIC | 1,920 | 0.616 s | 0.321 ms | 1.198 ms | 2.44% |
| `TransposeBatchMatMul` | AI_CORE | 1,920 | 0.420 s | 0.219 ms | 0.314 ms | 1.66% |
| `ScatterNdUpdateSk` | AI_VECTOR_CORE | 2,496 | 0.385 s | 0.154 ms | 0.303 ms | 1.52% |
| `_prepare_indexer_indices_kernel` | AI_VECTOR_CORE | 384 | 0.240 s | 0.626 ms | 0.963 ms | 0.95% |
| `_quantize_indexer_query_kernel` | AI_VECTOR_CORE | 384 | 0.113 s | 0.295 ms | 0.473 ms | 0.45% |
| `_engram_host_uva_gather_dequant_kernel` | AI_VECTOR_CORE | 78 | 0.070 s | 0.894 ms | 1.181 ms | 0.28% |

### 索引器的 pipe 比率（`kernel_top.csv`）

```
aic_mac_ratio    = 0.277      ← 27.7% 的算力真的在做 MAC
aic_mte1_ratio   = 0.514
aic_mte2_ratio   = 0.208
aiv_vec_ratio    = 0.915
cube_utilization = 97.6%
```

对比 sglang 侧同位置测得：**`aic_mac_ratio = 1.6%`、`cube_utilization = 99%`、96% 时间在 per-tile candidate mask 重扫**
（见 `KERNEL_CODE_MAP.md` §5.3 与 `sparse_lightning_indexer_service_vector_arch22.h:326-334` 的自述注释）。
**同一个模型、同一个算子、两种实现，MAC 占比差 17×。**

---

## 9. 与 sglang 逐项对账（每 4096-token chunk）

| 项 | sglang（BF16，ll / LIV2 + SLI） | **vLLM-Ascend（INT8 QLI V2）** | 倍数 |
|---|---|---|---|
| **索引器（同 S 单次 index-source）** | **984 ms** @ S=122,880 | **48.4 ms** @ S=122,880（实测） | **≈20.3×** |
| **索引器（整段 prefill / token 归一）** | 49–50 s / 64K token = **773 us/token** | 5.52 s / 127K token = **43 us/token** | **≈18×** |
| 索引器（每 chunk，晚段实测） | 3.94 s（4 次 × 984 ms） | **0.274 s**（chunk 29：1+4+3 次） | **≈14×** |
| 索引器（每 chunk，全段平均） | — | 0.172 s（5496 ms / 32） | — |
| MLA | 0.149 s | **0.150 s** | 1.0× |
| mHC | 0.35–0.40 s（纯 torch：`mix_stats_torch` + fp32 `post_torch`） | **0.122 s**（`HcPre` 0.078 + `HcPost` 0.044，融合算子） | ≈3× |
| 通信 | 未单独统计（TP 内 allreduce 为主） | 0.256 s（alltoallv + allreduce + allgather，其中 4.7 s/32 未重叠） | — |
| 其它（gemm/norm/moe/quant/other） | ~0.9 s | ~0.30 s | — |
| **每 chunk 墙钟** | **4.6 s** | **0.85 s** | **≈5.4×** |

### 9.1 关于倍数的口径（两次更正）

1. 早期草稿的 **"23×"** 是拿 sglang 的**晚段**单次成本除以 vLLM 的**全段平均**每 chunk 成本 —— 两个端点不在同一 S，**不能用**；
2. 之后的 **"8.8×"** 是拿 vLLM 在 S=54,347 的单次成本线性外推到 122,880 —— 但实测曲线**不是线性的**（见 §9.2），外推偏高；
3. **现在用实测**：vLLM 在 S=122,880 的单次 index-source 是 **48.4 ms**，sglang 同 S 是 **984 ms** ⇒ **20.3×**（同一 chunk 30：S=126,976，49.5 ms，同 S 外层推 984×126976/122880≈1017 ms ⇒ 20.5×，一致）。

### 9.2 索引器单次成本随 S 的实测曲线（vLLM，`va_calls.csv` 还原）

trace 覆盖了全部 32 个 chunk 的索引器调用（合计 5496 ms / op_statistic 5518 ms）。**每个 chunk 固定 8 次调用**：
**1 个 leading 调用 + 4 个完全相同的 index-source 调用（= 4 个索引源层）+ 3 个半维（K 的 head_dim=64）调用**。

| chunk | S≈ | leading (ms) | 4×index-source 单次 | 3×半维 单次 | chunk 合计 |
|---|---|---|---|---|---|
| 0 | 4,096 | 2.2 | 4.0 | 0.7 | 20.2 |
| 4 | 20,480 | 6.1 | 24.8 | 2.6 | 113.1 |
| 8 | 36,864 | 11.1 | 27.8 | 4.4 | 135.8 |
| 15 | 65,536 | 20.0 | 34.0 | 7.7 | 178.8 |
| 22 | 94,208 | 29.1 | 41.6 | 11.0 | 228.2 |
| 29 | 122,880 | 37.8 | **48.4** | 14.3 | 274.2 |
| 30 | 126,976 | 39.1 | **49.5** | 14.8 | 281.4 |
| 31 | 127,246 (T=270) | 3.4 | 4.3 | 1.4 | 24.9 |

曲线是**凹的**（不是纯线性）：
- S < 24K：约 **1.3 us/token**（4.0 → 24.8 ms）
- S > 24K：约 **0.24 us/token**（24.8 → 49.5 ms）

⇒ 说明 vLLM 的扫描量在 S 增大后有"封顶"效应（候选块数不再线性增长）。

**与 sglang 对比**：sglang 的单点 `984 ms @ S=122,880` = **8.0 us/token**，
即 vLLM 的**晚段斜率（0.24 us/token）比它低约 33×**，而 vLLM 的截距只有 2–3 ms。
⇒ **差距在斜率，不在截距** —— 也就是"**每个 key 的处理成本**"，而不是"固定开销"。
这与 `aic_mac_ratio` 的对比（vLLM 0.277 vs sglang 0.016，17.3×）互相印证：
sglang 的 96% 时间花在 per-tile candidate mask 重扫上，**每扫一个 key 都在重复读 mask**。

### 一句话归因### 一句话归因

> **vLLM 快，几乎全部来自索引器。**
> 每 chunk 索引器省下 `4.00 − 0.172 = 3.83 s`，而两边每 chunk 的差是 `4.6 − 0.85 = 3.75 s` —— **差额基本被这一项解释完**。
> MLA 一模一样（149 vs 150 ms）、mHC 快 3×（融合算子）、通信反而是 vLLM 更重（17% 未重叠）。

这**外部独立验证了**我们 sglang 侧的归因：
**"984 ms/次里 96% 花在 per-tile candidate mask 重扫，只有 1.6% 在做 MAC"** —— 换成 INT8 + 更好的实现后，同一个算子只需要 49.5 ms（max）。

---

## 10. 对 sglang 侧的三条可执行结论

1. **索引器是唯一值得大动的大头**：同 S 单次 984 ms → 目标 ~110 ms（vLLM 水平），按 token 归一就是从 773 us/token 降到 43 us/token；TTFT 73.7 s 有望降到 **20 s 量级**。
   路径：INT8 量化（`QuantSparseLightningIndexer` 只注册 arch22，而 sglang 的 `quant_*_dsl` 被 `is_npu_arch35()` 挡住 —— **门禁过严**）+ mask 重扫的合并（block table 一次读取 + Brcb）。
2. **mHC 融合算子可移植**：vLLM 的 `HcPre`/`HcPost` 在 A3（`ascend910_93`）上确实在用、比 sglang 的纯 torch 快 3×。
   收益 ≈ 0.25 s/chunk（TTFT 里约 4 s），且**不用换框架**。
3. **索引器修好后，通信会成为下一个瓶颈**：vLLM 的 17.3% 未重叠通信（4.71 s / 27.19 s）是"索引器变快之后"的必然结果。
   sglang 侧做 dp4+TP4 的通信规划时，可以直接用这个比例做预估。

## 11. 一个待解的小问题

vLLM 的索引器**每个 chunk 被调用 12 次**（384 = 32 × 12），而 sglang 侧是 **4 次**（4 个 index-source 层）。
可能原因：vLLM 把 4 个 source + 8 个 consumer 都走同一个 native op，而 sglang 的 consumer 走的是另一条（`LightningIndexerV2` / cfilter）路径。
若要进一步压缩，这一点值得看 —— 但它已经不是主要成本（12 × 14.4 ms = 173 ms/chunk）。

---

## 12. 复现清单（脚本都在 vllm-ascend 仓 `dsv41-perf` 分支的 `_tmp_preflight/`）

| 脚本 | 用途 |
|---|---|
| `collect_va_env.sh` / `collect_fail_in_container.sh` / `collect_va_failure.sh` | 环境版本、树差异、权重、flags、custom-ops 清单；失败现场（vllm 日志 + ascend device/plog + worker env） |
| `serve_b.sh` | 容器内起服务（含 25.5.x 的两个 workaround + fail-fast 自检；26.1.1 上不需要） |
| `bench_and_profile.sh` / `bench_client.sh` | sglang 同款 client 打 `/v1/completions`：run1 灌前缀 → run2 测 TTFT → run3 带 profiler |
| `collect_va_profile.py` | 从 `ASCEND_PROFILER_OUTPUT` 抽 <300 KB 证据包（桶/步/top kernel），或从 chrome trace 回退解析 |
| `PROFILE_SOP_VA.md` | profiling 三步 SOP（小样本验证 → 完整 → 后几轮） |
| `complete_image_tree.sh` | 把镜像里 fork 缺的 54 个 .py（TBE 算子源码）补进 fork 并做全量内容 diff |
