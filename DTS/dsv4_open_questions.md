# DSV4-Flash 学习：遗留问题清单

> 分支 `feat/dsv4-trace`（打点分支，只用于学习，**不合主干**）。
> 行号会漂：定位一律用 `grep -n`，不要背行号。
> 每条都写清三件事：**问题 / 已知证据 / 怎么验 + 判据**。

## Q1（2026-10-30）V4 的 SWA KV 池是"全局池 + 映射表"还是"每请求 128 的 ring"？

**问题**：`cr=0` 的层只看最近 128 个 token，那么这 128 个 token 的 KV 是
①从全量池里按窗口读，②一张独立的、按比例缩小的池子 + 映射表，③还是每个请求一个
128 槽位的 ring（新的顶掉旧的）？

**已知证据（代码）**

| 证据 | 位置 | 说明 |
|---|---|---|
| `swa_tokens = full_tokens * swa_full_tokens_ratio` | `mem_cache/pool_configurator.py` 的 hybrid SWA 分支 | V4 默认 `ratio=0.1`（`arg_groups/model_overrides/deepseek_v4.py`） |
| `translate_loc_from_full_to_swa()` 查表 | `mem_cache/swa_memory_pool.py` | SWA 侧是**独立数组**，不是全量池的视图 |
| "Per-request SWA ring … capacity is bounded by req slots" | `mem_cache/allocator/swa.py` 的 `_swa_req_ring` 分支 | 确实存在每请求 ring 的实现，但只在 DSPARK draft ring / V4.1 encoder replay 等路径启用 |
| `Initialize DeepSeekV4TokenToKVPool with max_num_reqs=… swa_size=…` | `mem_cache/deepseek_v4_memory_pool.py` 的 `DeepSeekV4TokenToKVPool.__init__` | 真实容量**启动时本来就打印了** |

**本次新增的打点**（`SGLANG_DSV4_TRACE=1` 时写 `<DIR>/pool.log`，一行一个事件）

| tag | 关键字段 | 想确认什么 |
|---|---|---|
| `v4_token_pool` | `swa_size / c4_size / c128_size / page_size / window / layers / cr_hist` | 每类池子到底分到多少 token；`cr_hist` 是 `[0,n] [4,n] [128,n]` |
| `swa_allocator` | `size_full / size_swa / ratio / req_ring / map_head` | `map_head=[0,1,2…]` ⇒ identity 映射；`req_ring=True` ⇒ 每请求 ring |
| `npu_v4_pool` | `c128_page_size / req_ring / request_window / paged_swa_alloc` | NPU 这套配置实际走哪种寻址 |

**怎么验**（不需要发请求，起服务即可）

```bash
export SGLANG_DSV4_TRACE=1
export SGLANG_DSV4_TRACE_DIR=/tmp/dsv4_trace
rm -rf /tmp/dsv4_trace
# 照常起服务，等 "server is fired up" 后：
cat /tmp/dsv4_trace/pool.log
```

**判据**

1. `ratio` 是否 ≈ 0.1 → 独立小池 vs 全量池；
2. `req_ring` 是 True 还是 False → 是否每请求 ring；
3. `map_head` 是否 identity → 槽位是"同一套编号"还是"压缩过的第二套编号"；
4. 与启动日志的 `Initialize DeepSeekV4TokenToKVPool with …` 数字对齐。

**现状**：待实测。

---

## 其他待实测/待确认

| # | 问题 | 现状 |
|---|---|---|
| 1 | `comb` 是否严格双随机（列和=1）；目前只打了 16 个值中的前 8 个 | 需把 `vals(comb, 8)` 改成 16 |
| 2 | decode 时 `compress_out` 从 `[0x512]` 变 `[1x512]` 的**回填语义**（写进哪个块、是否覆盖未满块） | 假设，待实测 |
| 3 | `sum(expert_tokens) == rows` 是否恒成立（目前 4/4 样本） | 强证据，未证 |
| 4 | decode 的 `9216 == 16 × 96 × 6` 与 `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 是否严格成正比 | 算术吻合，待实测（改成 48 应得 4608） |
| 5 | DSPARK 的结构与调用点 | 未讲 |

## 已确认（避免重复讨论）

- 43 层：`cr=0,0`（L00/L01）+ 之后 `4/128` 严格交替；只有 `cr=4` 带 indexer；`hash=True` 仅前 3 层；本 checkpoint 无 engram。
- 并行布局：`tp=16, attn_tp=1, attn_dp=16, moe_tp=1, moe_ep=16`；
  `tp = attn_dp × attn_tp = moe_ep × moe_dp × moe_tp`。
- MoE：prefill 用 `DeepEPNormal`（变长、`sum(expert_tokens)==rows`），decode 用 `DeepEPLL`（9216 定长 + `masked_m`）；shared expert 本地稠密、不进 A2A，最终 `1.5×routed + shared`。
- mHC：`R [T,4,4096]` 整层保持 4 流；每层两个子层各一次 `hc_pre`（收拢 + 预测 pre/post/comb）+ 一次 `hc_post`（`combᵀ·R + post⊙a`）；`comb` 在 pre 阶段做 softmax + 20 次 Sinkhorn。
