# Call stacks: 算子 → Python 调用栈

> 由 `_tmp_preflight/dump_callstack.py` 生成，用于把 profiler 里的 kernel 对回 sglang 代码。
> 数据源是 `trace_view.json` 的 Python 栈事件；`kernel_details.csv` 里**没有**栈。

| 项 | 值 |
|---|---|
| 输入 | `/tmp/demo_ep_late_1124_135951/node-61-47-19-68_447253_20261124140430132_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json` (745.3 MB) |
| 生成时间 | 2026-11-25T07:38:33+0000 |
| host / git | `node-61-47-19-68` / `33bdc88f80` |
| 事件数 / 带栈事件数 | 1141811 / 149967 (13.1%) |
| 目标清单 | default（21 条） |
| 关联 id→栈 表 | 0 |
| python span / 时间兜底 | 0 / 关（`--infer-by-time` 可开） |

**读法**：

- 每种栈按「最内层在前」打印；`launcher` = 最内层 `python/sglang/` 帧（拿不到时退回最内层任意仓库帧），
  也就是**该算子实际被哪一行 sglang 代码发起**，可直接回填 `KERNEL_CODE_MAP.md` 的「Python 发起位置」列。
- `仓库帧链` 是从内往外的完整 sglang/deep_ep 调用链（`A ⟵ B` 读作「A 被 B 调用」）。
- 栈方向按 torch 的 dispatch 帧位置自动判定；判不出来时按原始顺序打印，并在下面标注。
- 每个算子最多抽 `--per-op` 个事件、打印 `--stacks` 种栈，**是抽样不是全量**；次数列仍来自全量扫描。

## 汇总

| 算子 | 命中 | 带栈 | 合计 ms | 去重栈 | launcher（最内层 sglang 帧） |
|---|---|---|---|---|---|
| `SparseLightningIndexer` | 50 | 8 | 8006.4 | 1 | `.../attention/ascend_dsv4_backend.py:1755  _dsv41_index_qsli_topk` |
| `LightningIndexerV2` | 46 | 6 | 333.0 | 1 | `.../attention/ascend_dsv4_backend.py:1611  _dsv41_index_full_topk` |
| `SparseFlashMla` | 400 | 86 | 314.5 | 2 | `.../attention/ascend_dsv4_backend.py:2214  _forward_low_ratio_attention` |
| `SparseFlashMlaMetadata` | 36 | 6 | 5.5 | 1 | `.../attention/ascend_dsv4_backend.py:2074  _dsv41_attention_kernel_metadata` |
| `CamMoeCombineNormal` | 400 | 80 | 160.8 | 1 | `.../token_dispatcher/deepep.py:723  _combine_core` |
| `CamMoeDispatchNormal` | 400 | 80 | 66.4 | 1 | `.../token_dispatcher/deepep.py:668  _dispatch_core` |
| `NotifyDispatch` | 400 | 80 | 70.3 | 1 | `.../token_dispatcher/deepep.py:668  _dispatch_core` |
| `DispatchLayout` | 400 | 80 | 14.2 | 1 | `.../token_dispatcher/deepep.py:648  _dispatch_core` |
| `_hc_broadcast_kernel` | 8 | 2 | 0.9 | 1 | `.../layernorm/mhc.py:1863  hc_broadcast` |
| `LeftShift` | 36 | 4 | 40.1 | 1 | `.../layers/engram.py:122  _e8m0_to_float` |
| `RepeatInterleave` | 10 | 2 | 1.2 | 1 | `.../layers/engram.py:439  forward` |
| `BatchMatMulV2` | 240 | 0 | 377.6 | 0 | — |
| `MatMulV3` | 400 | 0 | 164.2 | 0 | — |
| `GroupedMatmul` | 400 | 160 | 29.0 | 2 | `.../moe/matmul.py:41  forward` |
| `QuantBatchMatmulV3` | 256 | 0 | 45.1 | 0 | — |
| `GatherV2` | 2 | 0 | 0.2 | 0 | — |
| `Cast` | 400 | 0 | 52.3 | 0 | — |
| `ReduceSum` | 400 | 0 | 10.0 | 0 | — |
| `RealDiv` | 400 | 0 | 5.1 | 0 | — |
| `Add/Adds` | 400 | 4 | 27.8 | 2 | `.../attention/ascend_dsv4_backend.py:3646  _stable_compact_1d` |
| `Mul` | 400 | 278 | 211.4 | 9 | `.../dsv4/dsv4_rope.py:165  apply_rotary_mul_inplace` |

## 逐算子

### `SparseLightningIndexer` — 命中 50 / 带栈 8，合计 8006.4 ms，1 种栈

**栈 #1 ×8**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../attention/ascend_dsv4_backend.py:1755  _dsv41_index_qsli_topk`

仓库帧链：`.../utils/arg_check.py:524  wrapper ⟵ .../sparse_lightning_indexer/sparse_lightning_indexer.py:197  sparse_lightning_indexer ⟵ .../indexer/sparse_lightning_indexer.py:39  _cann_impl ⟵ .../indexer/sparse_lightning_indexer.py:165  sparse_lightning_indexer ⟵ …(共 31 个仓库帧)`

```
← .../utils/arg_check.py:524  wrapper
← .../sparse_lightning_indexer/sparse_lightning_indexer.py:197  sparse_lightning_indexer
← .../indexer/sparse_lightning_indexer.py:39  _cann_impl
← .../indexer/sparse_lightning_indexer.py:165  sparse_lightning_indexer
← .../attention/ascend_dsv4_backend.py:1755  _dsv41_index_qsli_topk
← .../attention/ascend_dsv4_backend.py:1443  _low_ratio_index_topk
← .../attention/ascend_dsv4_backend.py:1338  forward_low_ratio_sources
← .../models/deepseek_v4.py:2626  _forward_prepare
← .../models/deepseek_v4.py:2910  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4_platform_stage.py:162  run_attn_hc
← …（共 41 帧，省略最外层 29 帧）
方向: 未确定（按原始顺序打印）
```

### `LightningIndexerV2` — 命中 46 / 带栈 6，合计 333.0 ms，1 种栈

**栈 #1 ×6**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../attention/ascend_dsv4_backend.py:1611  _dsv41_index_full_topk`

仓库帧链：`.../utils/arg_check.py:524  wrapper ⟵ .../lightning_indexer_v2/lightning_indexer.py:346  _lightning_indexer_npu ⟵ .../lightning_indexer_v2/lightning_indexer.py:455  lightning_indexer ⟵ .../indexer/lightning_indexer_v2.py:37  _cann_impl ⟵ …(共 32 个仓库帧)`

```
← .../utils/arg_check.py:524  wrapper
← .../lightning_indexer_v2/lightning_indexer.py:346  _lightning_indexer_npu
← .../lightning_indexer_v2/lightning_indexer.py:455  lightning_indexer
← .../indexer/lightning_indexer_v2.py:37  _cann_impl
← .../indexer/lightning_indexer_v2.py:151  lightning_indexer_v2
← .../attention/ascend_dsv4_backend.py:1611  _dsv41_index_full_topk
← .../attention/ascend_dsv4_backend.py:1445  _low_ratio_index_topk
← .../attention/ascend_dsv4_backend.py:1338  forward_low_ratio_sources
← .../models/deepseek_v4.py:2626  _forward_prepare
← .../models/deepseek_v4.py:2910  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← …（共 42 帧，省略最外层 30 帧）
方向: 未确定（按原始顺序打印）
```

### `SparseFlashMla` — 命中 400 / 带栈 86，合计 314.5 ms，2 种栈

**栈 #1 ×80**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../attention/ascend_dsv4_backend.py:2214  _forward_low_ratio_attention`

仓库帧链：`.../utils/arg_check.py:524  wrapper ⟵ .../sparse_flash_mla/sparse_flash_mla.py:352  sparse_flash_mla ⟵ .../attention/sparse_flash_mla.py:59  _cann_sparse_flash_mla ⟵ .../attention/sparse_flash_mla.py:34  _auto_sparse_flash_mla ⟵ …(共 30 个仓库帧)`

```
← .../utils/arg_check.py:524  wrapper
← .../sparse_flash_mla/sparse_flash_mla.py:352  sparse_flash_mla
← .../attention/sparse_flash_mla.py:59  _cann_sparse_flash_mla
← .../attention/sparse_flash_mla.py:34  _auto_sparse_flash_mla
← .../attention/sparse_flash_mla.py:207  sparse_flash_mla
← .../attention/ascend_dsv4_backend.py:2214  _forward_low_ratio_attention
← .../attention/ascend_dsv4_backend.py:4698  forward
← .../models/deepseek_v4.py:3010  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4_platform_stage.py:162  run_attn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← …（共 40 帧，省略最外层 28 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #2 ×6**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../attention/ascend_dsv4_backend.py:2074  _dsv41_attention_kernel_metadata`

仓库帧链：`.../utils/arg_check.py:524  wrapper ⟵ .../sparse_flash_mla/sparse_flash_mla.py:220  sparse_flash_mla_metadata ⟵ .../attention/sparse_flash_mla_metadata.py:73  _cann_sparse_flash_mla_metadata ⟵ .../attention/sparse_flash_mla_metadata.py:48  _auto_sparse_flash_mla_metadata ⟵ …(共 24 个仓库帧)`

```
← .../utils/arg_check.py:524  wrapper
← .../sparse_flash_mla/sparse_flash_mla.py:220  sparse_flash_mla_metadata
← .../attention/sparse_flash_mla_metadata.py:73  _cann_sparse_flash_mla_metadata
← .../attention/sparse_flash_mla_metadata.py:48  _auto_sparse_flash_mla_metadata
← .../attention/sparse_flash_mla_metadata.py:192  sparse_flash_mla_metadata
← .../attention/ascend_dsv4_backend.py:2074  _dsv41_attention_kernel_metadata
← .../attention/ascend_dsv4_backend.py:4527  _compute_kernel_metadata
← .../attention/ascend_dsv4_backend.py:4433  init_forward_metadata
← .../runner/eager_runner.py:306  _execute_extend
← .../runner/eager_runner.py:217  execute
← .../model_executor/model_runner.py:2191  _forward_raw
← .../model_executor/model_runner.py:1966  forward
← …（共 30 帧，省略最外层 18 帧）
方向: 未确定（按原始顺序打印）
```

### `SparseFlashMlaMetadata` — 命中 36 / 带栈 6，合计 5.5 ms，1 种栈

**栈 #1 ×6**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../attention/ascend_dsv4_backend.py:2074  _dsv41_attention_kernel_metadata`

仓库帧链：`.../utils/arg_check.py:524  wrapper ⟵ .../sparse_flash_mla/sparse_flash_mla.py:220  sparse_flash_mla_metadata ⟵ .../attention/sparse_flash_mla_metadata.py:73  _cann_sparse_flash_mla_metadata ⟵ .../attention/sparse_flash_mla_metadata.py:48  _auto_sparse_flash_mla_metadata ⟵ …(共 24 个仓库帧)`

```
← .../utils/arg_check.py:524  wrapper
← .../sparse_flash_mla/sparse_flash_mla.py:220  sparse_flash_mla_metadata
← .../attention/sparse_flash_mla_metadata.py:73  _cann_sparse_flash_mla_metadata
← .../attention/sparse_flash_mla_metadata.py:48  _auto_sparse_flash_mla_metadata
← .../attention/sparse_flash_mla_metadata.py:192  sparse_flash_mla_metadata
← .../attention/ascend_dsv4_backend.py:2074  _dsv41_attention_kernel_metadata
← .../attention/ascend_dsv4_backend.py:4527  _compute_kernel_metadata
← .../attention/ascend_dsv4_backend.py:4433  init_forward_metadata
← .../runner/eager_runner.py:306  _execute_extend
← .../runner/eager_runner.py:217  execute
← .../model_executor/model_runner.py:2191  _forward_raw
← .../model_executor/model_runner.py:1966  forward
← …（共 30 帧，省略最外层 18 帧）
方向: 未确定（按原始顺序打印）
```

### `CamMoeCombineNormal` — 命中 400 / 带栈 80，合计 160.8 ms，1 种栈

**栈 #1 ×80**  [cpu_op]  dur 0.1 / 0.1 / 0.2 ms (min/med/max)  launcher: `.../token_dispatcher/deepep.py:723  _combine_core`

仓库帧链：`.../strategies/normal_strategy.py:398  _intranode_combine ⟵ .../strategies/normal_strategy.py:365  combine ⟵ .../deep_ep/buffer.py:536  combine ⟵ .../deep_ep/utils.py:130  wrapper ⟵ …(共 38 个仓库帧)`

```
← .../strategies/normal_strategy.py:398  _intranode_combine
← .../strategies/normal_strategy.py:365  combine
← .../deep_ep/buffer.py:536  combine
← .../deep_ep/utils.py:130  wrapper
← .../token_dispatcher/deepep.py:723  _combine_core
← .../token_dispatcher/deepep.py:714  combine_b
← .../token_dispatcher/deepep.py:1101  combine_b
← .../token_dispatcher/deepep.py:1081  combine
← .../batch_overlap/two_batch_overlap.py:1193  _execute
← .../batch_overlap/two_batch_overlap.py:1205  combine
← .../fused_moe_triton/layer.py:1782  forward_impl
← .../ep_moe/layer.py:245  forward_impl
← …（共 50 帧，省略最外层 38 帧）
方向: 未确定（按原始顺序打印）
```

### `CamMoeDispatchNormal` — 命中 400 / 带栈 80，合计 66.4 ms，1 种栈

**栈 #1 ×80**  [cpu_op]  dur 0.1 / 0.1 / 0.2 ms (min/med/max)  launcher: `.../token_dispatcher/deepep.py:668  _dispatch_core`

仓库帧链：`.../strategies/normal_strategy.py:203  _intranode_dispatch ⟵ .../strategies/normal_strategy.py:120  dispatch ⟵ .../deep_ep/buffer.py:390  dispatch ⟵ .../deep_ep/utils.py:130  wrapper ⟵ …(共 39 个仓库帧)`

```
← .../strategies/normal_strategy.py:203  _intranode_dispatch
← .../strategies/normal_strategy.py:120  dispatch
← .../deep_ep/buffer.py:390  dispatch
← .../deep_ep/utils.py:130  wrapper
← .../token_dispatcher/deepep.py:668  _dispatch_core
← .../token_dispatcher/deepep.py:618  dispatch_b
← .../token_dispatcher/deepep.py:1074  dispatch_b
← .../token_dispatcher/deepep.py:1055  dispatch
← .../batch_overlap/two_batch_overlap.py:1193  _execute
← .../batch_overlap/two_batch_overlap.py:1196  dispatch
← .../fused_moe_triton/layer.py:1800  _dispatch_with_pre_quant
← .../fused_moe_triton/layer.py:1768  forward_impl
← …（共 51 帧，省略最外层 39 帧）
方向: 未确定（按原始顺序打印）
```

### `NotifyDispatch` — 命中 400 / 带栈 80，合计 70.3 ms，1 种栈

**栈 #1 ×80**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../token_dispatcher/deepep.py:668  _dispatch_core`

仓库帧链：`.../strategies/normal_strategy.py:203  _intranode_dispatch ⟵ .../strategies/normal_strategy.py:120  dispatch ⟵ .../deep_ep/buffer.py:390  dispatch ⟵ .../deep_ep/utils.py:130  wrapper ⟵ …(共 39 个仓库帧)`

```
← .../strategies/normal_strategy.py:203  _intranode_dispatch
← .../strategies/normal_strategy.py:120  dispatch
← .../deep_ep/buffer.py:390  dispatch
← .../deep_ep/utils.py:130  wrapper
← .../token_dispatcher/deepep.py:668  _dispatch_core
← .../token_dispatcher/deepep.py:618  dispatch_b
← .../token_dispatcher/deepep.py:1074  dispatch_b
← .../token_dispatcher/deepep.py:1055  dispatch
← .../batch_overlap/two_batch_overlap.py:1193  _execute
← .../batch_overlap/two_batch_overlap.py:1196  dispatch
← .../fused_moe_triton/layer.py:1800  _dispatch_with_pre_quant
← .../fused_moe_triton/layer.py:1768  forward_impl
← …（共 51 帧，省略最外层 39 帧）
方向: 未确定（按原始顺序打印）
```

### `DispatchLayout` — 命中 400 / 带栈 80，合计 14.2 ms，1 种栈

**栈 #1 ×80**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../token_dispatcher/deepep.py:648  _dispatch_core`

仓库帧链：`.../strategies/normal_strategy.py:60  get_dispatch_layout ⟵ .../deep_ep/buffer.py:258  get_dispatch_layout ⟵ .../token_dispatcher/deepep.py:648  _dispatch_core ⟵ .../token_dispatcher/deepep.py:618  dispatch_b ⟵ …(共 37 个仓库帧)`

```
← .../strategies/normal_strategy.py:60  get_dispatch_layout
← .../deep_ep/buffer.py:258  get_dispatch_layout
← .../token_dispatcher/deepep.py:648  _dispatch_core
← .../token_dispatcher/deepep.py:618  dispatch_b
← .../token_dispatcher/deepep.py:1074  dispatch_b
← .../token_dispatcher/deepep.py:1055  dispatch
← .../batch_overlap/two_batch_overlap.py:1193  _execute
← .../batch_overlap/two_batch_overlap.py:1196  dispatch
← .../fused_moe_triton/layer.py:1800  _dispatch_with_pre_quant
← .../fused_moe_triton/layer.py:1768  forward_impl
← .../ep_moe/layer.py:245  forward_impl
← .../ep_moe/layer.py:236  forward
← …（共 49 帧，省略最外层 37 帧）
方向: 未确定（按原始顺序打印）
```

### `_hc_broadcast_kernel` — 命中 8 / 带栈 2，合计 0.9 ms，1 种栈

**栈 #1 ×2**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../layernorm/mhc.py:1863  hc_broadcast`

仓库帧链：`.../layernorm/mhc.py:1863  hc_broadcast ⟵ .../models/deepseek_v4.py:4631  _forward_impl ⟵ .../models/deepseek_v4.py:4602  forward ⟵ .../models/deepseek_v4.py:5173  forward ⟵ …(共 20 个仓库帧)`

```
← .../ascend/driver.py:241  __call__
← .../runtime/jit.py:789  _run_jit_tail
← .../runtime/jit.py:728  run
← .../runtime/jit.py:370  <lambda>
← .../layernorm/mhc.py:1863  hc_broadcast
← .../models/deepseek_v4.py:4631  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← .../runner/eager_runner.py:339  _execute_extend
← .../runner/eager_runner.py:217  execute
← …（共 32 帧，省略最外层 20 帧）
方向: 未确定（按原始顺序打印）
```

### `LeftShift` — 命中 36 / 带栈 4，合计 40.1 ms，1 种栈

**栈 #1 ×4**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../layers/engram.py:122  _e8m0_to_float`

仓库帧链：`.../layers/engram.py:122  _e8m0_to_float ⟵ .../layers/engram.py:1581  _owned_rows ⟵ .../layers/engram.py:1716  _dp_global_buffer_lookup ⟵ .../layers/engram.py:1620  _dp_sharded_lookup ⟵ …(共 27 个仓库帧)`

```
← .../layers/engram.py:122  _e8m0_to_float
← .../layers/engram.py:1581  _owned_rows
← .../layers/engram.py:1716  _dp_global_buffer_lookup
← .../layers/engram.py:1620  _dp_sharded_lookup
← .../layers/engram.py:1510  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../layers/engram.py:1899  embed_rows
← .../layers/engram.py:1868  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4.py:4381  _forward_layers_hc_pre_from_prev
← …（共 39 帧，省略最外层 27 帧）
方向: 未确定（按原始顺序打印）
```

### `RepeatInterleave` — 命中 10 / 带栈 2，合计 1.2 ms，1 种栈

**栈 #1 ×2**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../layers/engram.py:439  forward`

仓库帧链：`.../layers/engram.py:439  forward ⟵ .../models/deepseek_v4.py:4203  _forward_layers_hc_pre_from_prev ⟵ .../models/deepseek_v4.py:4702  _forward_impl ⟵ .../models/deepseek_v4.py:4602  forward ⟵ …(共 21 个仓库帧)`

```
← .../layers/engram.py:439  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4.py:4203  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← .../runner/eager_runner.py:339  _execute_extend
← .../runner/eager_runner.py:217  execute
← .../model_executor/model_runner.py:2191  _forward_raw
← …（共 31 帧，省略最外层 19 帧）
方向: 未确定（按原始顺序打印）
```

### `BatchMatMulV2` — 命中 240 / 带栈 0，合计 377.6 ms，0 种栈

> [!] 这 240 个事件没有栈，也没有可关联（`External id`/`correlation`）的带栈事件：这类 `cat` 本身不携带栈。

### `MatMulV3` — 命中 400 / 带栈 0，合计 164.2 ms，0 种栈

> [!] 这 400 个事件没有栈，也没有可关联（`External id`/`correlation`）的带栈事件：这类 `cat` 本身不携带栈。

### `GroupedMatmul` — 命中 400 / 带栈 160，合计 29.0 ms，2 种栈

**栈 #1 ×80**  [cpu_op]  dur 0.1 / 0.1 / 0.3 ms (min/med/max)  launcher: `.../moe/matmul.py:41  forward`

仓库帧链：`.../moe/matmul.py:41  forward ⟵ .../quantization/moe_methods.py:841  apply ⟵ .../moe_runner/ascend.py:310  run ⟵ .../moe_runner/runner.py:254  run ⟵ …(共 35 个仓库帧)`

```
← .../torch/_ops.py:1279  __call__
← .../moe/matmul.py:41  forward
← .../quantization/moe_methods.py:841  apply
← .../moe_runner/ascend.py:310  run
← .../moe_runner/runner.py:254  run
← .../modelslim/modelslim.py:1157  apply
← .../fused_moe_triton/layer.py:1837  run_moe_core
← .../ep_moe/layer.py:272  run_moe_core
← .../fused_moe_triton/layer.py:1772  forward_impl
← .../ep_moe/layer.py:245  forward_impl
← .../ep_moe/layer.py:236  forward
← .../modules/module.py:1789  _call_impl
← …（共 48 帧，省略最外层 36 帧）
方向: inner-first（本来就是最内层在前）
```

**栈 #2 ×80**  [cpu_op]  dur 0.1 / 0.1 / 0.2 ms (min/med/max)  launcher: `.../moe/matmul.py:41  forward`

仓库帧链：`.../moe/matmul.py:41  forward ⟵ .../quantization/moe_methods.py:841  apply ⟵ .../moe_runner/ascend.py:278  run ⟵ .../moe_runner/runner.py:254  run ⟵ …(共 35 个仓库帧)`

```
← .../torch/_ops.py:1279  __call__
← .../moe/matmul.py:41  forward
← .../quantization/moe_methods.py:841  apply
← .../moe_runner/ascend.py:278  run
← .../moe_runner/runner.py:254  run
← .../modelslim/modelslim.py:1157  apply
← .../fused_moe_triton/layer.py:1837  run_moe_core
← .../ep_moe/layer.py:272  run_moe_core
← .../fused_moe_triton/layer.py:1772  forward_impl
← .../ep_moe/layer.py:245  forward_impl
← .../ep_moe/layer.py:236  forward
← .../modules/module.py:1789  _call_impl
← …（共 48 帧，省略最外层 36 帧）
方向: inner-first（本来就是最内层在前）
```

### `QuantBatchMatmulV3` — 命中 256 / 带栈 0，合计 45.1 ms，0 种栈

> [!] 这 256 个事件没有栈，也没有可关联（`External id`/`correlation`）的带栈事件：这类 `cat` 本身不携带栈。

### `GatherV2` — 命中 2 / 带栈 0，合计 0.2 ms，0 种栈

> [!] 这 2 个事件没有栈，也没有可关联（`External id`/`correlation`）的带栈事件：这类 `cat` 本身不携带栈。

### `Cast` — 命中 400 / 带栈 0，合计 52.3 ms，0 种栈

> [!] 这 400 个事件没有栈，也没有可关联（`External id`/`correlation`）的带栈事件：这类 `cat` 本身不携带栈。

### `ReduceSum` — 命中 400 / 带栈 0，合计 10.0 ms，0 种栈

> [!] 这 400 个事件没有栈，也没有可关联（`External id`/`correlation`）的带栈事件：这类 `cat` 本身不携带栈。

### `RealDiv` — 命中 400 / 带栈 0，合计 5.1 ms，0 种栈

> [!] 这 400 个事件没有栈，也没有可关联（`External id`/`correlation`）的带栈事件：这类 `cat` 本身不携带栈。

### `Add/Adds` — 命中 400 / 带栈 4，合计 27.8 ms，2 种栈

**栈 #1 ×2**  [cpu_op]  dur 0.3 / 0.3 / 0.3 ms (min/med/max)  launcher: `.../attention/ascend_dsv4_backend.py:3646  _stable_compact_1d`

仓库帧链：`.../attention/ascend_dsv4_backend.py:3646  _stable_compact_1d ⟵ .../attention/ascend_dsv4_backend.py:2412  _build_dsv41_ratio_metadata ⟵ .../attention/ascend_dsv4_backend.py:2340  _build_low_ratio_compress_metadata ⟵ .../attention/ascend_dsv4_backend.py:418  _build_npu_compress_metadata ⟵ …(共 21 个仓库帧)`

```
← .../attention/ascend_dsv4_backend.py:3646  _stable_compact_1d
← .../attention/ascend_dsv4_backend.py:2412  _build_dsv41_ratio_metadata
← .../attention/ascend_dsv4_backend.py:2340  _build_low_ratio_compress_metadata
← .../attention/ascend_dsv4_backend.py:418  _build_npu_compress_metadata
← .../attention/ascend_dsv4_backend.py:4436  init_forward_metadata
← .../runner/eager_runner.py:306  _execute_extend
← .../runner/eager_runner.py:217  execute
← .../model_executor/model_runner.py:2191  _forward_raw
← .../model_executor/model_runner.py:1966  forward
← .../managers/tp_worker.py:689  forward_batch_generation
← .../dspark_components/dspark_worker_v2.py:847  _forward_prefill
← .../dspark_components/dspark_worker_v2.py:798  forward_batch_generation
← …（共 27 帧，省略最外层 15 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #2 ×2**  [cpu_op]  dur 0.2 / 0.2 / 0.2 ms (min/med/max)  launcher: `.../attention/ascend_dsv4_backend.py:3646  _stable_compact_1d`

仓库帧链：`.../attention/ascend_dsv4_backend.py:3646  _stable_compact_1d ⟵ .../attention/ascend_dsv4_backend.py:2411  _build_dsv41_ratio_metadata ⟵ .../attention/ascend_dsv4_backend.py:2340  _build_low_ratio_compress_metadata ⟵ .../attention/ascend_dsv4_backend.py:418  _build_npu_compress_metadata ⟵ …(共 21 个仓库帧)`

```
← .../attention/ascend_dsv4_backend.py:3646  _stable_compact_1d
← .../attention/ascend_dsv4_backend.py:2411  _build_dsv41_ratio_metadata
← .../attention/ascend_dsv4_backend.py:2340  _build_low_ratio_compress_metadata
← .../attention/ascend_dsv4_backend.py:418  _build_npu_compress_metadata
← .../attention/ascend_dsv4_backend.py:4436  init_forward_metadata
← .../runner/eager_runner.py:306  _execute_extend
← .../runner/eager_runner.py:217  execute
← .../model_executor/model_runner.py:2191  _forward_raw
← .../model_executor/model_runner.py:1966  forward
← .../managers/tp_worker.py:689  forward_batch_generation
← .../dspark_components/dspark_worker_v2.py:847  _forward_prefill
← .../dspark_components/dspark_worker_v2.py:798  forward_batch_generation
← …（共 27 帧，省略最外层 15 帧）
方向: 未确定（按原始顺序打印）
```

### `Mul` — 命中 400 / 带栈 278，合计 211.4 ms，9 种栈

**栈 #1 ×80**  [cpu_op]  dur 0.5 / 0.5 / 0.9 ms (min/med/max)  launcher: `.../dsv4/dsv4_rope.py:165  apply_rotary_mul_inplace`

仓库帧链：`.../inplace_partial_rotary_mul/inplace_partial_rotary_mul.py:156  inplace_partial_rotary_mul ⟵ .../rope/inplace_partial_rotary_mul.py:42  _cann_ops_transformer_inplace_partial_rotary_mul ⟵ .../rope/inplace_partial_rotary_mul.py:24  _auto_inplace_partial_rotary_mul ⟵ .../rope/inplace_partial_rotary_mul.py:84  inplace_partial_rotary_mul ⟵ …(共 29 个仓库帧)`

```
← .../torch/_ops.py:1279  __call__
← .../inplace_partial_rotary_mul/inplace_partial_rotary_mul.py:156  inplace_partial_rotary_mul
← .../rope/inplace_partial_rotary_mul.py:42  _cann_ops_transformer_inplace_partial_rotary_mul
← .../rope/inplace_partial_rotary_mul.py:24  _auto_inplace_partial_rotary_mul
← .../rope/inplace_partial_rotary_mul.py:84  inplace_partial_rotary_mul
← .../dsv4/dsv4_rope.py:165  apply_rotary_mul_inplace
← .../models/deepseek_v4.py:2522  _forward_prepare
← .../models/deepseek_v4.py:2910  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4_platform_stage.py:162  run_attn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← …（共 40 帧，省略最外层 28 帧）
方向: inner-first（本来就是最内层在前）
```

**栈 #2 ×80**  [cpu_op]  dur 0.4 / 0.5 / 0.7 ms (min/med/max)  launcher: `.../dsv4/dsv4_rope.py:165  apply_rotary_mul_inplace`

仓库帧链：`.../inplace_partial_rotary_mul/inplace_partial_rotary_mul.py:156  inplace_partial_rotary_mul ⟵ .../rope/inplace_partial_rotary_mul.py:42  _cann_ops_transformer_inplace_partial_rotary_mul ⟵ .../rope/inplace_partial_rotary_mul.py:24  _auto_inplace_partial_rotary_mul ⟵ .../rope/inplace_partial_rotary_mul.py:84  inplace_partial_rotary_mul ⟵ …(共 29 个仓库帧)`

```
← .../torch/_ops.py:1279  __call__
← .../inplace_partial_rotary_mul/inplace_partial_rotary_mul.py:156  inplace_partial_rotary_mul
← .../rope/inplace_partial_rotary_mul.py:42  _cann_ops_transformer_inplace_partial_rotary_mul
← .../rope/inplace_partial_rotary_mul.py:24  _auto_inplace_partial_rotary_mul
← .../rope/inplace_partial_rotary_mul.py:84  inplace_partial_rotary_mul
← .../dsv4/dsv4_rope.py:165  apply_rotary_mul_inplace
← .../models/deepseek_v4.py:3081  _project_wo_a
← .../models/deepseek_v4.py:3023  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4_platform_stage.py:162  run_attn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← …（共 40 帧，省略最外层 28 帧）
方向: inner-first（本来就是最内层在前）
```

**栈 #3 ×80**  [cpu_op]  dur 0.4 / 0.5 / 0.7 ms (min/med/max)  launcher: `.../dsv4/dsv4_rope.py:177  apply_rotary_mul_inplace`

仓库帧链：`.../inplace_partial_rotary_mul/inplace_partial_rotary_mul.py:156  inplace_partial_rotary_mul ⟵ .../rope/inplace_partial_rotary_mul.py:42  _cann_ops_transformer_inplace_partial_rotary_mul ⟵ .../rope/inplace_partial_rotary_mul.py:24  _auto_inplace_partial_rotary_mul ⟵ .../rope/inplace_partial_rotary_mul.py:84  inplace_partial_rotary_mul ⟵ …(共 29 个仓库帧)`

```
← .../torch/_ops.py:1279  __call__
← .../inplace_partial_rotary_mul/inplace_partial_rotary_mul.py:156  inplace_partial_rotary_mul
← .../rope/inplace_partial_rotary_mul.py:42  _cann_ops_transformer_inplace_partial_rotary_mul
← .../rope/inplace_partial_rotary_mul.py:24  _auto_inplace_partial_rotary_mul
← .../rope/inplace_partial_rotary_mul.py:84  inplace_partial_rotary_mul
← .../dsv4/dsv4_rope.py:177  apply_rotary_mul_inplace
← .../models/deepseek_v4.py:2522  _forward_prepare
← .../models/deepseek_v4.py:2910  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4_platform_stage.py:162  run_attn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← …（共 40 帧，省略最外层 28 帧）
方向: inner-first（本来就是最内层在前）
```

**栈 #4 ×8**  [cpu_op]  dur 0.5 / 0.5 / 0.7 ms (min/med/max)  launcher: `.../dsv4/dsv4_rope.py:165  apply_rotary_mul_inplace`

仓库帧链：`.../inplace_partial_rotary_mul/inplace_partial_rotary_mul.py:156  inplace_partial_rotary_mul ⟵ .../rope/inplace_partial_rotary_mul.py:42  _cann_ops_transformer_inplace_partial_rotary_mul ⟵ .../rope/inplace_partial_rotary_mul.py:24  _auto_inplace_partial_rotary_mul ⟵ .../rope/inplace_partial_rotary_mul.py:84  inplace_partial_rotary_mul ⟵ …(共 32 个仓库帧)`

```
← .../torch/_ops.py:1279  __call__
← .../inplace_partial_rotary_mul/inplace_partial_rotary_mul.py:156  inplace_partial_rotary_mul
← .../rope/inplace_partial_rotary_mul.py:42  _cann_ops_transformer_inplace_partial_rotary_mul
← .../rope/inplace_partial_rotary_mul.py:24  _auto_inplace_partial_rotary_mul
← .../rope/inplace_partial_rotary_mul.py:84  inplace_partial_rotary_mul
← .../dsv4/dsv4_rope.py:165  apply_rotary_mul_inplace
← .../attention/ascend_dsv4_backend.py:1494  _dsv41_index_k_store
← .../attention/ascend_dsv4_backend.py:1434  _low_ratio_index_topk
← .../attention/ascend_dsv4_backend.py:1338  forward_low_ratio_sources
← .../models/deepseek_v4.py:2626  _forward_prepare
← .../models/deepseek_v4.py:2910  forward
← .../modules/module.py:1789  _call_impl
← …（共 43 帧，省略最外层 31 帧）
方向: inner-first（本来就是最内层在前）
```

_（另有 5 种较少见的栈未打印，`--stacks` 可调）_

## 回填用速查（可直接贴进 KERNEL_CODE_MAP.md 的映射表）

| 算子 | 本次 trace 次数 | launcher（Python 发起位置） |
|---|---|---|
| `SparseLightningIndexer` | 50 | `.../attention/ascend_dsv4_backend.py:1755  _dsv41_index_qsli_topk` |
| `LightningIndexerV2` | 46 | `.../attention/ascend_dsv4_backend.py:1611  _dsv41_index_full_topk` |
| `SparseFlashMla` | 400 | `.../attention/ascend_dsv4_backend.py:2214  _forward_low_ratio_attention` |
| `SparseFlashMlaMetadata` | 36 | `.../attention/ascend_dsv4_backend.py:2074  _dsv41_attention_kernel_metadata` |
| `CamMoeCombineNormal` | 400 | `.../token_dispatcher/deepep.py:723  _combine_core` |
| `CamMoeDispatchNormal` | 400 | `.../token_dispatcher/deepep.py:668  _dispatch_core` |
| `NotifyDispatch` | 400 | `.../token_dispatcher/deepep.py:668  _dispatch_core` |
| `DispatchLayout` | 400 | `.../token_dispatcher/deepep.py:648  _dispatch_core` |
| `_hc_broadcast_kernel` | 8 | `.../layernorm/mhc.py:1863  hc_broadcast` |
| `LeftShift` | 36 | `.../layers/engram.py:122  _e8m0_to_float` |
| `RepeatInterleave` | 10 | `.../layers/engram.py:439  forward` |
| `BatchMatMulV2` | 240 | （无栈，未验证） |
| `MatMulV3` | 400 | （无栈，未验证） |
| `GroupedMatmul` | 400 | `.../moe/matmul.py:41  forward` |
| `QuantBatchMatmulV3` | 256 | （无栈，未验证） |
| `GatherV2` | 2 | （无栈，未验证） |
| `Cast` | 400 | （无栈，未验证） |
| `ReduceSum` | 400 | （无栈，未验证） |
| `RealDiv` | 400 | （无栈，未验证） |
| `Add/Adds` | 400 | `.../attention/ascend_dsv4_backend.py:3646  _stable_compact_1d` |
| `Mul` | 400 | `.../dsv4/dsv4_rope.py:165  apply_rotary_mul_inplace` |
