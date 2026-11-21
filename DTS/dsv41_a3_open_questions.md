# DSV4.1-Flash / Ascend A3 学习：遗留问题清单

> 打点分支 `feat/dsv41-a3-trace`（从 PR #50205 的 `pr-50205-dsv41-a3-w8a8` 切出，**只用于学习，永不合并**）。
> 行号会漂：定位一律用 `grep -n`，不要背行号。
> 方法见 `dsv4-trace-method` skill；上一代（V4 / A5）的记录在 `feat/dsv4-trace` 的 `DTS/dsv4_open_questions.md`。
>
> 环境变量一律 `SGLANG_DSV41_TRACE*`，日志默认 `/tmp/dsv41_trace`，默认**关闭**。

## 0. 本分支已加的打点（按批次，每批只跑一次）

| 批次 | tag | 状态 | 调用点 |
|---|---|---|---|
| ① 全局 | `model_in` / `model_out` | ✅ 已加（本文件对应这一轮） | `models/deepseek_v4.py::DeepseekV4Model._forward_impl` |
| ② 层 | `layer_in` / `layer_out` / `ffn_in` / `ffn_out` | ⏳ 待加 | `models/deepseek_v4_platform_stage.py::NonCudaDecoderMixin.forward` |
| ③ 注意力 | `attn_in` / `attn_out` / `attn_backend` / `compress_out` / `indexer_topk` | ⏳ 待加 | `models/deepseek_v4.py::MQALayer.*` + `hardware_backend/npu/attention/ascend_dsv4_backend.py` |
| ④ MoE / 池 | `moe_dispatch_*` / `moe_runner_out` / `moe_out` / `pool_*` | ⏳ 待加 | `layers/moe/fused_moe_triton/layer.py`、`moe_runner/ascend.py`、`dsv4_memory_pool.py` |

`model_out` 有 3 个出口，用 `stage=` 区分：`pp`（非末级 PP）、`dspark`（投机捕获）、`final`（常规返回）。
这是本分支与 `feat/dsv4-trace` 的差异：PR 版 `_forward_impl` 多了 `hc_pre_from_prev` 和 TBO 两条主干，
模型级事件必须落在"最终出口"这一层，不能在层循环里打。

## 0.5 基线实测（2026-11-21，环境修复后首次跑通）

镜像 `sgl_kernel_npu 2026.11.0.post4` + 补装 `custom_ops 2026.11.0.post3`
（`_tmp_preflight/FIX_SUMMARY.md` 有完整过程）。启动耗时：

```
load_weight=111.73  kv_cache_allocation=0.87  scheduler_e2e=328.93
cuda_graph={prefill=0.00, decode=0.00, target_verify=191.09, draft_prefill=0.00, draft_decode=10.13}
```

- `target_verify=191.09s` 正是补装前崩掉的那条路径（DSpark 定宽 verify 图），现在能捕获；
- prefill 图被自动禁用（日志里 `npu graph: False`），**所以 prefill 走 eager、打点不受图影响**；
- 8 个 attn-DP rank 各自收到同一个 warmup prefill（`#new-token: 6`）。

## 1. 本轮（批次①）运行方式

用包装脚本，不必改 `run_dsv41.sh`：

```bash
cd /home/a00821909/sglang
bash _tmp_preflight/run_trace_batch1.sh      # 内部 source vendor + 导出 trace env + 跑 run_dsv41.sh
```

它等价于：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
rm -rf /tmp/dsv41_trace
SGLANG_DSV41_TRACE=1 SGLANG_DSV41_TRACE_DIR=/tmp/dsv41_trace \
SGLANG_DSV41_TRACE_RANKS=0 SGLANG_DSV41_TRACE_LIMIT=0  bash run_dsv41.sh
```

为什么这一轮**不用**关图：prefill 本来就是 eager（见 §0.5），`model_in/model_out` 在 prefill 里
一定能打出来；decode 若走图 replay 则不再经过 Python，日志不会爆炸。

注意点：

- `LIMIT=0`：warmup、图捕获、你的请求都会记。**区分方法**：`grep "model_in" ... | tail -3`，
  你那条请求的 `T` 通常明显大于 warmup 的 6；`--disable-radix-cache` 你已经开了（H8 不成问题）；
- 只发**一个**请求，`max_new_tokens` 给小（8 左右）；
- 想看更干净可以在 `run_dsv41.sh` 里加 `--skip-server-warmup`（H7），非必需。

## 2. 先预测，后看日志

| # | 预测 | 依据 | 判据 |
|---|---|---|---|
| P1 | `r00.log` 里 `model_in` **恰好 1 条** | `LIMIT=1` + 只发一次请求 | >1 条 ⇒ warmup 或 radix 没关 |
| P2 | `model_in R=3d[T,4,4096]bf16` | `_forward_impl` 入口后 `hc_broadcast(x, hc_mult)` | 3d 但第二维 ≠ 4 ⇒ `hc_mult` 不是 4 |
| P3 | `T` == prompt token 数（未分片全量） | `hc_broadcast` 作用在本 rank 的 `hidden_states` 上 | `T` ≈ prompt/16 ⇒ attn-DP 已切片 |
| P4 | `sl=0 el=43`，`tp=16`、`attn_dp=8`（按真实启动参数 `--tp-size 16 --dp-size 8` 修正）⇒ 预测 `attn_tp=2`、`moe_tp=1`、`moe_ep=16` | PR 的 A3 布局 + 日志里 `DP0 TP0 … DP7 TP15` = 8×2 | `attn_tp` 不是 2、或 `moe_ep` 不是 16 ⇒ 我对并行组的理解有偏差 |
| P5 | `a2a=MoeA2ABackend.DEEPEP` | `--moe-a2a-backend deepep` | 打印别的 ⇒ 参数没生效/被覆盖 |
| P6 | `model_out` 只有 `stage=final` 一条，`y=2d[T,4096]` | 无 PP、未开 DSpark 捕获 | 出现 `stage=dspark` ⇒ 服务端开了投机 |

**预测错了就明说**，把实测数字写进下面的 Q 条目，并注明证伪了哪条预测。

## Q1 `model_in` 看到的 `T` 到底是全量 token 还是本 attn-DP rank 的分片？

**已知证据**：`_forward_impl` 入口（`grep -n "def _forward_impl" python/sglang/srt/models/deepseek_v4.py`）先做
`hc_broadcast`，之后才做 `input_ids` 的 DP gather（`dp_gather_replicate` 分支），说明入口处 `T` 是本 rank 视角。

**怎么验**：`head -2 /tmp/dsv41_trace/r00.log` + 与启动日志里的 prompt token 数对比。

**判据**：`T` == 全 prompt ⇒ 未切片（attn-DP gather 在上层完成）；`T` ≈ prompt/16 ⇒ 本 rank 切片。

**现状**：待实测。

## Q2 A3 的 `hc_mult` 与层数是否与 PR 描述的 43 层 / 4 流一致？

**已知证据**：PR 的 `deepseek_v4_mhc.py` 增加 A3 torch 版 `mix_stats_torch`/`post_torch`；
`_kernel_num_heads` 在 `_is_npu_a3` 时不 padding 到 64（`grep -n "_kernel_num_heads" python/sglang/srt/models/deepseek_v4.py`）。

**怎么验**：`grep -o "hc_mult=[0-9]*" /tmp/dsv41_trace/r00.log | head -1` 与 `el - sl`。

**判据**：`hc_mult=4` 且 `el-sl=43` ⇒ 与 V4 同构；否则 A3 换了残差结构。

**现状**：待实测。

## Q3 这条 A3 路径真正走的是哪个注意力后端（C1/C2 分流点在哪）？

**已知证据**：PR 新增 `LowRatioCompressAscendBackendMixin`（`ascend_dsv4_backend.py:1300`），
`forward_core` 里拒绝跨代 ratio；A3 走 `sparse_flash_mla` / `lightning_indexer_v2`。

**怎么验**：批次③的 `attn_backend` tag（下一轮加），一行看出 `kind=`。

**判据**：`kind=swa` 的层应只有前 2 层（若 `compress_ratios[0:2]=[0,0]`）；
`kind=compressed1/2` 与 `cr=1/2` 的层数应完全对应。

**现状**：待实测（需要批次③）。

## Q4 DSpark 的 draft forward 会不会让 `model_out` 多出 `stage=dspark` 行？

**已知证据**：`_forward_impl` 里 `capture_dspark = self.dspark_layers_to_capture is not None`，
为真时返回 `(hidden_states, pre_hc_head)` 并带 aux 捕获。

**怎么验**：开 `--speculative-algorithm` 跑一次，看 `grep "stage=dspark"` 是否有行。

**判据**：有 ⇒ 该 checkpoint 的投机走同一 `_forward_impl` 的捕获分支；没有 ⇒ draft 走另一入口。

**现状**：待实测（本轮不开投机）。
