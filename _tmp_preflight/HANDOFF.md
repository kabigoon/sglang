# 交接文档：DeepSeek-V4.1-Flash (A3/NPU) 128K 性能分析与优化

> 写给下一个上下文窗口。先读完这份，再动手。
> 所有脚本都在 `_tmp_preflight/`（**临时运维目录，不属于 PR 内容，不要合进主干**）。
> 分支 `pr-50205-dsv41-a3-w8a8`（PR #50205 的代码 + 运维脚本），`feat/dsv41-a3-trace`（同一个 + 打点代码）。

---

## 1. 任务目标

在 **昇腾 A3 × 16 die** 上跑 **DeepSeek-V4.1-Flash**，达标线：

| 项 | 目标 | 现状 | 判定 |
|---|---|---|---|
| 输入/输出 | **128K / 1K** | 129,091 / 1,024 | ✅ |
| 前缀缓存命中率 | **50%**（固定，不可改） | 实测 55%（含 warmup 全命中条，见 §6） | ✅ |
| **TTFT** | **< 30 s** | **74.9 s**（median，128K/55% 命中） | ❌ 差 2.5× |
| **TPOT** | **< 50 ms** | **32.8 ms**（开 DSpark） | ✅ 余量 34% |

**唯一未达标的是 TTFT。** 64K 输入同配置是 **25.9s → 达标**，说明环境健康，问题在长上下文 prefill 效率。

---

## 2. 环境（已修复，别再折腾）

### 2.1 硬件/软件
- 容器内 root，仓库 `/home/a00821909/sglang`（git 挂载），服务端口 **5684**
- 权重：`/home/weights/DeepSeek-V4.1-Flash-1017-w4a8`（493G，**在用**）、`...-w8a8`（728G，精度对照用）
- `torch 2.13.0` / `torch_npu 2.13.0` / `CANN 9.2.0` / `sgl_kernel_npu 2026.11.0.post4`（新代镜像，无 custom-ops）
- 两个 checkpoint 都是 `quant_method: "ascend"`，描述文件 `quant_model_description.json`，scheme `W4A8_DYNAMIC` / `W8A8_DYNAMIC`

### 2.2 关键修复（一次性的，已固化在当前容器里）
新镜像从 `sgl-kernel-npu 2026.11.0.post4` 起**不再打包 custom-ops/ops-transformer**，而 PR 代码还在用 13 个 `torch.ops.custom.*` → 首次 forward 报 `AttributeError`。

修法（走完即生效，容器重启需重做）：下载 post3 的 `custom-ops` + `ops-transformer` zip，装进 `/usr/local/Ascend/cann-9.2.0/opp`，再 `pip install custom_ops-1.0-*.whl`。

```bash
LOCAL_DIR=$PWD/_tmp_preflight/zips bash _tmp_preflight/fix_install_custom_ops.sh
bash _tmp_preflight/check_image.sh     # 期望 OPS_CUSTOM=13/13, IDEAL=YES
bash _tmp_preflight/check_import.sh    # 期望 OPS_AFTER_INIT=10/10
```

细节见 `_tmp_preflight/FIX_SUMMARY.md`（含 sha256、代理 SSL 拦截处理、import 顺序坑）。

### 2.3 启动前必须做的事
1. `source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/{customize,custom_transformer}/bin/set_env.bash`
   （**不 source → `import custom_ops` 失败 → 又被 try/except 吞成 warning → forward 时才报 AttributeError**）
2. `import custom_ops` 必须在 `torch` / `torch_npu` **之后**
3. 启动脚本**不能有 `set -u`**（CANN 的 set_env.sh 引用未定义变量会静默中止整个脚本）

---

## 3. 启动命令与关键参数

### 3.1 标准启动（等价于最终形态）

```bash
cd /home/a00821909/sglang
DP_SIZE=8 CHUNKED_PREFILL=32768 MEM_FRACTION=0.85 DECODE_GRAPH=1 \
  bash _tmp_preflight/run_dsv41_w4a8.sh
```

脚本内部固定：`--tp-size 16 --enable-dp-attention --dp-size $DP_SIZE`、`--quantization` 不传（走 checkpoint 自带的 ascend 描述）、`--json-model-override-args '{"architectures": ["DeepseekV4ForCausalLM"], "quantization_config": null}'`、`--speculative-algorithm DSPARK`（EAGLE 不用）、`--page-size 128 --kv-cache-dtype auto --mem-fraction-static $MEM_FRACTION --max-prefill-tokens 131072`、`--language-model-only --reasoning-parser deepseek-v4`、**radix cache 开启**（原脚本里的 `--disable-radix-cache` 已去掉，否则命中率永远 0%）。

### 3.2 参数语义（每一条都踩过坑）

| 参数 | 语义 / 坑 |
|---|---|
| `DP_SIZE` | `--enable-dp-attention --dp-size N`；组内 `attn_tp = 16/N` |
| `CHUNKED_PREFILL` | **全局值**，sglang 会 `/attn_dp_size`。要每 rank 4096，全局必须写 `4096×DP_SIZE`。忘乘会变成 512/rank（254 个 chunk） |
| `MAX_RUNNING` | 全局值，同样被除（16/8=2 每 rank） |
| `MEM_FRACTION` | 权重+KV 预算。**开 DSpark 时必须 >0.817**，否则 `ValueError: Loaded weights leave no GPU memory`。0.85 时池子 ≈500K token |
| `DECODE_GRAPH` | `1` 默认。**dp≠8 时必须配 `DECODE_BS="1 2 4 8 16"`**，否则 `AssertionError: capture_bs=[]`（见 §7.3） |
| `PREFILL_GRAPH=1` | **别开**：`KeyError: 'block_tables'`，DSV4.1 的 prefill 路径不支持图 |
| `EXTRA_ENV` | 任意 `K=V` 透传（用于扫 `SGLANG_OPT_*` 开关） |
| `DISABLE_SPEC=1` | 关 DSpark（TTFT 不受影响，但 TPOT 变差；纯 prefill 调试用） |

### 3.3 输出给 PR 用的精度基线（已完成）
`bash _tmp_preflight/eval_gsm8k.sh` → **GSM8K 194/200 = 97%**（w8a8，200 条子集，greedy，thinking on，evalscope 1.15/math_verify 判分）。
**w4a8 的精度还没测**（改权重后应补一次，若 <90% 要警惕量化 scheme 不对）。

---

## 4. 脚本清单（`_tmp_preflight/`）

| 脚本 | 用途 |
|---|---|
| `run_dsv41_w4a8.sh` | **主启动器**。所有 env 见 §3.2；会打印配置摘要（含每 rank 实际值） |
| `run_dp4_debug.sh` | 单独起 server 的调试版（日志落盘 + 崩了给提示），默认 dp=4 |
| `oneshot_bench.sh` | **一次一条命令**：起 server → 等 ready → 跑 bench → 打「登记行」→ 停 server。机器不独占时用这个 |
| `bench_128k_1k.sh` | client 压测（128K/1K、50% 共享前缀、SLO 判定）。会打印完整 python 命令，便于登记/复现 |
| `bench_ladder.sh` | 并发阶梯（一次预热跑 C=1/2/4/8） |
| `sweep_ttft.sh` | 过夜扫参（配置矩阵 + 超时保护 + 自动出 report.md） |
| `eval_gsm8k.sh` / `analyze_eval.py` | 精度评测 / 结果分析（含 finish_reason、错题表） |
| `check_image.sh` / `check_import.sh` | 镜像算子代际检查 / 干跑 import（H15 检查） |
| `fix_install_custom_ops.sh` | 补装 post3 老算子包 |
| `find_artifacts.sh` / `dump_ops.sh` / `check_any_image.sh` | 离线找包 / 算子全量 dump / 宿主机检查任意镜像 |
| `run_trace_batch1.sh` | 打点批次①（在 `feat/dsv41-a3-trace` 分支；打点暂停中） |

### 4.1 最常用的三条

```bash
# ① 单次完整实验（推荐，机器不独占时）
DP_SIZE=4 CHUNKED_PREFILL=16384 MEM_FRACTION=0.85 DECODE_GRAPH=1 DECODE_BS="1 2 4 8 16" \
  WARMUP=4 NUM_PROMPTS=8 CONCURRENCY=1 CACHE_PCT=50 bash _tmp_preflight/oneshot_bench.sh

# ② server 已在跑，只跑 client
WARMUP=4 NUM_PROMPTS=8 CONCURRENCY=1 DP_SIZE=4 CACHE_PCT=50 zsh _tmp_preflight/bench_128k_1k.sh

# ③ 精度
bash _tmp_preflight/eval_gsm8k.sh && python3 _tmp_preflight/analyze_eval.py
```

### 4.2 client 侧参数（写登记表用）

```
输入 = --gsp-system-prompt-len 63130 + --gsp-question-len 63130（50% 共享）
      名义 126,260 → 实际 tokenize 后 ≈129,091（gsp 生成文本再 tokenize 会涨 2.2%）
输出 = --gsp-output-len 1024；--num-prompts 8；--max-concurrency 1；--seed 42
--warmup-requests = DP_SIZE  ← 必须覆盖所有 DP rank（前缀缓存按 rank 分区！）
--cache-report --output-details
```
`IN_LEN`/`MAX_SEQ`/`TOKEN_OVERSHOOT=1.03` 决定上面的 63130，别手填。

---

## 5. 已测数据（全部 dp=4 或 dp=8、C=1、spec=on、chunk=4096/rank）

### 5.1 dp=8（sweep 过夜，`/tmp/dsv41_sweep/report.md`）

| 配置 | TTFT mean | TPOT | HIT | 池子 | 备注 |
|---|---|---|---|---|---|
| baseline (chunk 4096/rank, mem .85) | 67,938 | 33.05 | 54.14% | 500,608 | **基准** |
| baseline_rep | 67,916 | 33.05 | 54.14% | 500,480 | 复现性 0.03% |
| chunk6144 | 67,494 | 34.48 | 54.14% | 443,392 | 噪声内 |
| chunk8192 | 70,556 | 33.09 | 51.36% | 387,712 | 更差 |
| mem087 / mem090 | 67,979 / 67,999 | 33.1 | 54.14% | 908K / 1.52M | 池子无关 |
| chunk12288 / 16384 | — | — | — | — | INCOMPLETE(OOM/超时) |
| dp4 / dp2 / prefill_graph | — | — | — | — | FAIL_NOT_READY（原因见 §7） |

### 5.2 dp=4（同机、单点精测）

| 场景 | 输入 | HIT | U(未命中) | TTFT mean | **TTFT median** | TPOT | 判定 |
|---|---|---|---|---|---|---|---|
| 128K / 0% | 129,099 | 0% | 129.1K | 110,738 | **110,769** | 32.3 | FAIL |
| 128K / 55% | 129,099 | 55.03% | 58.1K | 65,793 | **74,873** | 32.8 | FAIL |
| 64K / 55.6% | 66,991 | 55.65% | 29.7K | 22,804 | **25,930** | 24.6 | ✅ PASS |

**登记表用 median**：mean 被"warmup 全命中的那 1 条"拉低（128K 时 mean 65.8 vs median 74.9）。

---

## 6. 已排除的假设（别再重复试）

| 假设 | 结果 |
|---|---|
| "chunk 之间的调度开销是瓶颈"（我提的） | ❌ 证伪：4096/6144/8192 → 67.9/67.5/70.6s，加大反而更差 |
| "池子太大挤占激活内存" | ❌ 证伪：mem .85/.87/.90 → TTFT 几乎不变（67.9-68.0s） |
| "prefill 图能吃掉 launch 开销" | ❌ 死路：`KeyError: 'block_tables'` |
| "dp 降低（attn_tp 2→4）能提速" | ❌ 基本无效：mean 65.8s vs 67.9s（−3%），median 反而更高 |
| "TTFT ≈ 固定46s + 线性项"（我提的） | ❌ 证伪：用它预测 64K 会得 60.6s，实测 25.9s |

**当前正确的结论**：TTFT 对上下文**超线性**（2× 上下文 → 2.89× 时间），是 **长上下文 attention 的算力/带宽**问题；同命中率下 128K 的 prefill 效率只有 64K 的 68%（775 vs 1147 tok/s 等效速率）。128K/50% 要达标需 **2.5×** 提升。

---

## 7. 未闭合问题（下一个窗口可以接着查）

1. **96K 标度点未测**：平滑 T^1.5 还是 96K 附近有路径突变？
   `IN_LEN=98304 CACHE_PCT=50 WARMUP=4 NUM_PROMPTS=8 CONCURRENCY=1 DP_SIZE=4 bash _tmp_preflight/bench_128k_1k.sh`（不用重启）
2. **7 个被关掉的算子开关未试**：`SGLANG_OPT_FUSE_WQA_WKV=0`、`USE_FUSED_HASH_TOPK=False`、`USE_TILELANG_MHC_PRE/POST=False`、`DEEPGEMM_HC_PRENORM=False`、`FP8_WO_A_GEMM=False`、`SGLANG_DSV4_NATIVE_ATTN=0`、`SHALLOW_FUSED_ROPE_FQ4=0`。用 `EXTRA_ENV="SGLANG_OPT_FUSE_WQA_WKV=1"` 逐个试，**每次都要过 GSM8K**。
3. **profiler 没上**：`--profile` / torch profiler 定位 128K prefill 里 attention vs FFN vs compressor 占比。
4. **dp≠8 的 `capture_bs=[]` 机制未验证**：放宽 `--cuda-graph-bs-decode` 到 `1 2 4 8 16` 就好了，推断是 [base_cuda_graph_runner.py:112](../python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py) 的整除过滤（`bs*alignment_width % mul_base == 0`，DSpark 下 alignment_width=6），**但没证实**。验证：`grep -iE "capture" /tmp/dp4_*.log`。
5. **并发未测**：`bench_ladder.sh`（C=1/2/4/8）。注意 dp=8+mem0.85 时池子只有 500K → 最多 3 条 128K 并发；要测并发需 mem 0.90（池子 1.52M，TTFT 不受影响）。
6. **w4a8 精度未测**（w8a8 是 97%）。
7. **dp=8 的 median TTFT 没单独记**（只有 mean 67.9s），要严格对比 dp=4 需补一条。

---

## 8. 工作纪律（重要）

- **只改 `_tmp_preflight/`**，不要动 PR 代码（`python/sglang/**`）。提交信息带 `(not part of the PR)`。
- 每次实验**只改一个变量**；改完记进登记表（含 `decode graph` 状态，它影响 TPOT 可比性）。
- 机器**不独占**：用 `oneshot_bench.sh`（跑完自动停 server），别留 server 占卡。
- 单次实验 10-15 分钟（起 server 3-5 分钟 + warmup dp×110s + 8 条实测）。
- 报数用 **median**（见 §5.2 注）。
- 分支：`pr-50205-dsv41-a3-w8a8`（运维+PR 代码）、`feat/dsv41-a3-trace`（打点，暂停）。推完记得两边都更新。

---

## 9. 一句话状态

> 环境已修好、128K 能跑通、TPOT 已达标（33ms）、GSM8K 97%；**只剩 TTFT**（128K/50% 需 74.9s → 30s）。
> 已排除 chunk/mem/prefill-graph/dp 四个方向，确认瓶颈是长上下文 attention 的超线性成本。
> 下一步优先级：**96K 标度点 → 算子开关 → profiler**。
