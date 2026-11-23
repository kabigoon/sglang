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
| `oneshot_bench.sh` | **一次一条命令**：起 server → 等 ready → 跑 bench → 打「登记行」→ 停 server。机器不独占时用这个。现在支持 `SCALE="65536 98304 126250"`（同一 server 跑标度曲线）、`IN_LEN=`、`OUT_LEN=`、`EXTRA_ENV=`（扫开关）、`DECODE_BS=`、`DISABLE_SPEC=`、`SKIP_FLUSH=` 透传，并自动输出**标度表 + 段间标度指数 + 带 median 的登记行** |
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

# ①b 标度曲线：起一次 server 跑多个输入长度（省掉每次 3-5 分钟起服务；唯一变量=上下文长度）
DP_SIZE=4 CHUNKED_PREFILL=16384 MEM_FRACTION=0.85 DECODE_GRAPH=1 DECODE_BS="1 2 4 8 16" \
  WARMUP=4 NUM_PROMPTS=8 CONCURRENCY=1 CACHE_PCT=50 SCALE="65536 98304 126250" \
  bash _tmp_preflight/oneshot_bench.sh

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

### 5.3 标度的双变量拟合（本轮新增：只用 §5.2 已有 3 点，**没有新测量**）

把 TTFT 当成 **(T=总输入长度, U=未命中 token 数)** 的函数去拟合上面 3 点：

```
TTFT ≈ 6.93e-7 · T^1.11 · U^0.49      （3 点、2 参数，恰好定解）
  64K/55.6%  T=66,991  U=29,700  → 实测 25.93s   模型 25.93s
  128K/55%   T=129,099 U=58,100  → 实测 74.87s   模型 74.87s
  128K/0%    T=129,099 U=129,000 → 实测 110.77s  模型 110.77s
```

**这推翻了"超线性"的旧表述。** 旧结论说"2× 上下文 → 2.89× 时间"，但那条曲线里 T 和 U 是同时变大的，
把两个变量混在了一起。用 U 单变量幂律（指数 1.58，就是旧结论）去预测 128K/0% 会得 **264s**，实测 110.8s，**差 2.4×**。
双变量模型的三个指数含义：

| 项 | 指数 | 含义 |
|---|---|---|
| T（总长度） | **1.11** | 缓存命中的 token **并没有被跳过**：新 token 的 attention 仍要扫过全部 T 个 key。前缀缓存省的是 cached 段上的 FFN/linear，**不省 attention 对 cached prefix 的计算** |
| U（未命中） | **0.49** | 亚线性：未命中部分被 chunk（4096/rank）× DP rank 并行摊掉了 |
| 推论 | — | 128K/50% 的 2.5× 缺口**不可能靠加并行/chunk 拿到**（U 项只有 0.49 次幂，且 chunk 已证伪）；只能砍 T 项的常数或让 attention 不必扫全 prefix |

对下一步的三条影响：

1. **96K 标度点的预测值**：模型给 **53.2s**（不同拟合口径给 48–56s）。实测若落在这区间 → 平滑，问题在 kernel 常数因子；
   若明显偏低（例如 <40s）→ 说明 96K 附近有路径突变，那里的分支/kernel 选择就是首要线索。
2. **7 个开关的优先级要重排**：先打 **attention 路径**上的（`SGLANG_DSV4_NATIVE_ATTN`、`SGLANG_SHALLOW_FUSED_ROPE_FQ4`、
   `SGLANG_OPT_USE_TILELANG_MHC_PRE/POST`、compressor 相关），再打 FFN 路径上的（`FP8_WO_A_GEMM`、`DEEPGEMM_HC_PRENORM`）——
   按 T^1.11 的读数，FFN 只占长上下文 prefill 的次要部分。
3. **profiler 要按"每 chunk 的 attention 时间是否随 prefix 长度线性涨"来切**，这才是能直接证伪/证实 T^1.11 的解释。

⚠ 这是 3 点拟合出来的**假设**，不是结论；它比旧表述更能解释已有数据（尤其是 128K/0% 那点差 2.4×），但需要 96K 点和 profiler 去验。

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

> ⚠ **上面这段已被 §5.3 改写**（"2.89×"把 T 和 U 混在一起了；正确形式是 `T^1.11 · U^0.49`）。
> 保留原文只为对照。"是长上下文 attention 的问题"这个方向仍然成立，但机制从"超线性算力"变成
> "**attention 仍要扫过已缓存的 prefix**"，所以抓手也不同（不是加并行，是砍 prefix 扫描的常数/让 prefix 免扫）。

---

## 7. 未闭合问题（下一个窗口可以接着查）

1. **96K 标度点未测**（脚本已就绪，见 §10）。平滑 T^1.5 还是 96K 附近有路径突变？§5.3 给的点预测是 **53.2s**。
   ```bash
   DP_SIZE=4 CHUNKED_PREFILL=16384 MEM_FRACTION=0.85 DECODE_GRAPH=1 DECODE_BS="1 2 4 8 16" \
     WARMUP=4 NUM_PROMPTS=8 CONCURRENCY=1 CACHE_PCT=50 SCALE="98304" \
     bash _tmp_preflight/oneshot_bench.sh
   ```
   （`KEEP_SERVER=1` 可保留 server 接着跑别的长度，省一次启动）
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
> 已排除 chunk/mem/prefill-graph/dp 四个方向。瓶颈机制按 §5.3 改判为"**attention 仍要扫过已缓存的 prefix**"
> （TTFT ≈ `T^1.11 · U^0.49`），不是单纯"超线性算力"。
> 下一步优先级：**96K 标度点（命令已就绪，§7.1/§10）→ 算子开关（先 attention 路径）→ profiler**。

---

## 10. 本轮（接手窗口）的产出与待跑命令

本轮**没有新测量数据**（这台 Mac 上没有到 NPU 机器的通道，实验要靠人跑、贴回传块）。

### 10.1 改动清单（全在 `_tmp_preflight/`，不碰 PR 代码）

| 文件 | 改动 |
|---|---|
| `HANDOFF.md` | 新增 §5.3（双变量标度拟合，改写旧"超线性"结论）、§6 加对照注、§7.1 换成可跑命令、本节 |
| `oneshot_bench.sh` | 新增 `SCALE=`（同一 server 跑标度曲线）/`IN_LEN=`/`OUT_LEN=`/`EXTRA_ENV=`/`DECODE_BS=`/`DISABLE_SPEC=`/`SKIP_FLUSH=` 透传；新增**标度表 + 段间标度指数 + 带 median 的登记行**；修 `$?` 取到 tail 退出码的 bug（改 `PIPESTATUS[0]`）；透传 `EXTRA_ENV` 后 §7.2 的开关扫描才真正能一键跑 |
| `bench_128k_1k.sh` | 修 `${ACTUAL_IN}` 在定义前被 heredoc 引用（`set -u` 下 `cat` 直接失败，口径头段打印不出来）；把预估提前到 heredoc 之前 |
| `oneshot_bench.sh`（第 3 轮） | **改用后台跑 + 心跳**：原来 `\| tee \| tail -30`，`tail` 要等 EOF 才吐字，屏幕上会静默 10-15 分钟（"看不出跑没跑"就是这么来的）。现在每 `HB`（默认 30）秒打一行「已运行 Ns + 日志最后一行」，跑完再打末尾 30 行；开头打印 `脚本版本: <git sha>`，一眼看出机器上跑的是不是新脚本 |
| `oneshot_bench.sh`（第 3 轮） | **修 `$SLOG）` 这类 locale 相关的隐蔽崩溃**（见 §10.4 ②）：变量名后紧跟中文标点时改成 `${SLOG}` 写法，3 处 |

自测（无 NPU 也能做的部分）：`bash -n` 通过；`DRY_RUN=1` 输出正确；用 3 组合成 jsonl 跑通了抽数/表格/指数/登记行四段 python；用假 server（本地 5684 假 HTTP）+ 假 bench 搭了端到端 harness，验证了心跳、退出码、抽数失败降级（标 `ERR` 不中断循环）、server 清理、单点/标度两种输出。

### 10.2 待跑（按顺序，每轮只变一个变量）

```bash
cd /home/a00821909/sglang
git fetch origin && git log --oneline -1        # 确认在最新提交（脚本会自己打印版本 sha）
tmux new -s bench                               # 建议：SSH 断了不会连带杀掉 bench

# ① 96K 标度点：唯一变量=上下文长度（dp=4，与 §5.2 三点可比）
DP_SIZE=4 CHUNKED_PREFILL=16384 MEM_FRACTION=0.85 DECODE_GRAPH=1 DECODE_BS="1 2 4 8 16" \
  WARMUP=4 NUM_PROMPTS=8 CONCURRENCY=1 CACHE_PCT=50 SCALE="98304" \
  bash _tmp_preflight/oneshot_bench.sh

# ② 若 ① 落在 48-56s（平滑）→ 按 §5.3 的优先级扫开关，先 attention 路径：
DP_SIZE=4 CHUNKED_PREFILL=16384 MEM_FRACTION=0.85 DECODE_GRAPH=1 DECODE_BS="1 2 4 8 16" \
  WARMUP=4 NUM_PROMPTS=8 CONCURRENCY=1 CACHE_PCT=50 \
  EXTRA_ENV="SGLANG_DSV4_NATIVE_ATTN=1" bash _tmp_preflight/oneshot_bench.sh
#   过了 TTFT 关还要过精度：bash _tmp_preflight/eval_gsm8k.sh && python3 _tmp_preflight/analyze_eval.py
#   注意 SGLANG_OPT_FUSE_WQA_WKV 在 run 脚本里是 "=0" 而其它是 "=False"，EXTRA_ENV 里照抄原写法（K=V 直接 export）
```

### 10.3 回传要点（贴回来我就能接着判）

只需要 `===== 标度曲线（复制整段回传） =====` 到 `===== 回传块结束 =====` 之间那段：
含 `IN_LEN / actual_in / HIT% / U_unc / TTFT_mean / TTFT_med / TPOT / TPUT / U_per_s` 九个字段，以及自动算的**段间标度指数**。
另外带一句 server 侧的 `max_total_num_tokens`（脚本已自动打印）。

判据：指数平滑 → kernel 常数因子（走开关/profiler）；某段指数明显突跳 → 那里有路径突变，先查分支。

### 10.4 "看不见输出 / 不知道跑没跑"的排障（本轮新增）

症状：屏幕上只有前几行，之后十几分钟一个字都没有；或者完全没有结果。

| 可能 | 怎么确认 | 状态 |
|---|---|---|
| **① bench 的 stdout 被 `tail -30` 吞了**（旧版 `\| tee \| tail -30`，`tail` 等 EOF 才吐字） | `/tmp/oneshot_*/bench_in*.log` **在实时增长**：`tail -f` 能看到；`wc -l` 每隔半分钟变大 | ✅ 已改：后台跑 + 心跳 + 末尾 30 行 |
| **② 变量名后紧跟中文标点**（`"…（日志 $SLOG）"`）→ UTF-8 locale 下 bash 把标点字节算进变量名 → `set -u` 报 `SLOG?: unbound variable` **直接退出**，此时还没起 server | `locale`；日志/屏幕上搜 `unbound variable`。实测：`LC_ALL=C` 不报，`C.UTF-8` / `en_US.UTF-8` 必报 | ✅ 已改 `${SLOG}`（**这是原脚本就有的老 bug**，不是本轮引入的） |
| **③ 会话断了**（SSH 断 → 子进程被杀，输出随终端丢） | `pgrep -f sglang.launch_server` 为空但 `/tmp/oneshot_*` 目录存在 | 用 `tmux` 或 `nohup … > /tmp/run.log 2>&1 &` |

自检三连（贴回传块时一起带上，我就能判"跑到哪一步死的"）：

```bash
ls -lt /tmp/oneshot_* | head              # 有无本次目录（目录名带启动时刻）
ls -lt /home/dsv4_test/bench_out | head   # bench 自己落的 jsonl，跟我的脚本无关，一定有
cat $(ls -dt /tmp/oneshot_* | head -1)/summary.tsv      # 每点一行，抽数失败标 ERR
tail -40 $(ls -dt /tmp/oneshot_* | head -1)/server.log  # 起 server 阶段的问题都在这
```

注意 `/tmp/oneshot_*` 里会有**空目录**：脚本先 `mkdir` 再判断 `DRY_RUN`，所以干跑（以及任何在 `[2/3]` 之前就退出的情况）都会留下空壳——
**目录存在 ≠ 跑过**，要看里面有没有 `server.log` / `bench_in*.log`。

