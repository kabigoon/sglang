# DSV4-Flash 学习：遗留问题清单

> 分支 `feat/dsv4-trace`（打点分支，只用于学习，**不合主干**）。
> 行号会漂：定位一律用 `grep -n`，不要背行号。
> 每条都写清三件事：**问题 / 已知证据 / 怎么验 + 判据**。

## Q1 ✅ 已结案（2026-10-31）：V4 的 SWA KV 池怎么分配？

**原问题**：`cr=0` 的层只看最近 128 个 token，那么这 128 个 token 的 KV 是
①从全量池里按窗口读，②一张独立的、按比例缩小的池子 + 映射表，③还是每个请求一个
128 槽位的 ring（新的顶掉旧的）？

**结论：②。** 是**第二套独立的 KV 池（全量池的 10%）+ 一张动态映射表**：

- 每个**活着**的 token 在两套池里各占一个槽：`full_attn_allocator` 和
  `swa_attn_allocator` 各分配一次，然后写映射
  `full_to_swa_index_mapping[alloc_full] = alloc_swa`；
- 滑窗 128 **只限制读**（kernel 的 `ori_win_left = window - 1`），不决定分配；
- SWA 槽与全量槽**同生共死**：`free()` 里先 `free_swa()` 再 `free_full()`；
- 因此 SWA 池（只有 10% 容量）是**显存瓶颈侧**，调度器为它单独做淘汰
  （`reclaim_for_prealloc` → `EvictParams(swa_num_tokens=…)`）；
- "每请求 ring" 的实现**存在但未启用**（`req_ring=False`），只在 DSPARK draft ring /
  V4.1 encoder replay 等路径上开。

> 所以对"是不是只分配 128 个槽、新的顶旧的"这个问题：**不是**。窗口只影响读范围。

**实测数据**（16 die，`max_num_reqs=12`，`SGLANG_MOE_TRACE` 风格的 `<DIR>/pool.log`）

| 量 | 值 | 关系 |
|---|---|---|
| `size_full` | 1468928 | 全量 token 预算（≈1.47M） |
| `size_swa` | 146816 | `= size_full × 0.0999`，即 `swa_full_tokens_ratio=0.1` 生效 |
| `c4_size` | 367232 | `= size_full / 4` ✓ |
| `c128_size` | 11476 | `= ceil(size_full / 128)` ✓ |
| `c4_state` / `c128_state` | 9176 / 1664 | 压缩器状态池，sizing 规则未读（见 Q2） |
| `page_size` / `swa_page_size` | 128 / 128 | |
| `c128_page_size` | 16 | NPU 稀疏注意力算子要求 16 的倍数 |
| `window` / `layers` | 128 / 43 | |
| `req_ring` / `ring_cost` | False / 0 | 没走每请求 ring |
| `map_len` | 1469057 | `= size_full + page_size + 1`（末尾 -1 哨兵） |
| `map_head` | `[0,0,0,…]` | 初始化为 0、运行时逐 token 填（identity 会是 `[0,1,2,…]`） |
| `paged_swa_alloc` / `unified` | True / False | 走 paged allocator；非 unified KV |

**估算显存**（`--kv-cache-dtype bfloat16`，每 token 每层 512 维 × 2B = 1 KB）

| 池 | 估算 |
|---|---|
| SWA 146816 × 1KB × 43 层 | ≈ 6.3 GB |
| C4 367232 × 1KB × 21 层 | ≈ 7.7 GB |
| C128 11476 × 1KB × 20 层 | ≈ 0.23 GB |
| 合计 | ≈ 14.2 GB |

与 `--mem-fraction-static 0.68`（≈41 GB）+ W8A8 权重 ≈19 GB/卡 是自洽的。

**副产品（待验）**：`cr_hist=[[0,5],[4,21],[128,20]]` 共 **46** 项，但模型只有 **43** 层，
且运行时只有 L00/L01 两层 `cr=0`。差的 3 个 0 应该属于 **MTP/nextn 层**：checkpoint 的
`compress_ratios` 把 MTP 层也列了进去，而 SGLang 的 nextn 层用硬编码覆盖
`COMPRESS_RATIO_NEXTN_LAYER = 0`（`models/deepseek_v4_nextn.py`），并不读这个列表。

```bash
python3 -c "import json;c=json.load(open('/home/weights/DeepSeek-V4-Flash-0731-w8a8/config.json'));\
print(len(c['compress_ratios']), c.get('num_hidden_layers'), c.get('num_nextn_predict_layers'))"
# 期望：46 43 3
```

---

## Q2（2026-10-31 新）压缩器状态池的 sizing 规则是什么？

**问题**：`c4_state=9176`、`c128_state=1664` 是怎么算出来的？它们与
`swa_size(146816)`、`c4_size(367232)`、`num_req_slots(13)`、`window(128)` 是什么关系？

**已有线索**：`9176 = 146816 / 16` 恰好整除（可疑，但 `/16` 没有明显语义）；
`1664 / 13 = 128`（等于 window，也可能是巧合）。两个等式不能同时解释这两个数。

**怎么验**：读 `mem_cache/deepseek_v4_compress_state.py` 的 `CompressStatePool.__init__`、
以及 `hardware_backend/npu/dsv4/dsv4_memory_pool.py` 里 `_get_state_pool` 的构造；或换一个
`--max-running-requests`（例如 24）再起一次，看这两个数是否按 req 槽数线性变化。

**判据**：能写出 `c4_state = f(...)`、`c128_state = g(...)` 的闭式表达式，并用第二次实测验证。

---

## Q3（2026-10-31 新）`swa_full_tokens_ratio=0.1` 的依据是什么？

**问题**：这个 0.1 是经验值，还是能从 `max_num_reqs × window` 推出来？

**已有线索**：146816 tokens ≈ 1147 个 page（page=128），而 12 个请求若各留一个窗口
（128 token + 1 页余量）只需要 ≈ 24 页——**配额比"窗口所需"大 40 多倍**，说明它不是按窗口
算的，更可能与"淘汰压力 / 吞吐"有关（SWA 池是 `reclaim_for_prealloc` 的瓶颈侧）。

**怎么验**：`git log -S "swa_full_tokens_ratio" --oneline` 找引入它的提交与说明；或把 ratio
调到 0.05 / 0.2 各起一次，观察是否 OOM、是否频繁触发 SWA 淘汰。

---

## 其他待实测/待确认

| # | 问题 | 现状 |
|---|---|---|
| 1 | `comb` 是否严格双随机（列和=1）；目前只打了 16 个值中的前 8 个 | 需把 `vals(comb, 8)` 改成 16 |
| 2 | decode 时 `compress_out` 从 `[0x512]` 变 `[1x512]` 的**回填语义**（写进哪个块、是否覆盖未满块） | 假设，待实测 |
| 3 | `sum(expert_tokens) == rows` 是否恒成立（目前 4/4 样本） | 强证据，未证 |
| 4 | decode 的 `9216 == 16 × 96 × 6` 与 `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 是否严格成正比 | 算术吻合，待实测（改成 48 应得 4608） |
| 5 | DSPARK 的结构与调用点 | 未讲 |
| 6 | `len(compress_ratios)==46` 且 `num_nextn_predict_layers==3`（Q1 副产品） | 待验，一条命令 |

## 已确认（避免重复讨论）

- 43 层：`cr=0,0`（L00/L01）+ 之后 `4/128` 严格交替；只有 `cr=4` 带 indexer；
  `hash=True` 仅前 3 层；本 checkpoint 无 engram。
- 并行布局：`tp=16, attn_tp=1, attn_dp=16, moe_tp=1, moe_ep=16`；
  `tp = attn_dp × attn_tp = moe_ep × moe_dp × moe_tp`。
- MoE：prefill 用 `DeepEPNormal`（变长、`sum(expert_tokens)==rows`），
  decode 用 `DeepEPLL`（9216 定长 + `masked_m`）；shared expert 本地稠密、不进 A2A，
  最终 `1.5×routed + shared`。
- mHC：`R [T,4,4096]` 整层保持 4 流；每层两个子层各一次 `hc_pre`（收拢 + 预测
  pre/post/comb）+ 一次 `hc_post`（`combᵀ·R + post⊙a`）；`comb` 在 pre 阶段做
  softmax + 20 次 Sinkhorn。
- KV 池（Q1）：SWA / C4 / C128 是**三套独立池**，容量分别按 `1 : 1/4 : 1/128`
  从全量 token 预算折算；SWA 那套只占 10%，是容量瓶颈。
