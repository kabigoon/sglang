# Call stacks: 算子 → Python 调用栈

> 由 `_tmp_preflight/dump_callstack.py` 生成，用于把 profiler 里的 kernel 对回 sglang 代码。
> 数据源是 `trace_view.json` 的 Python 栈事件；`kernel_details.csv` 里**没有**栈。

| 项 | 值 |
|---|---|
| 输入 | `/tmp/demo_ep_late_1124_135951/node-61-47-19-68_447253_20261124140430132_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json` (745.3 MB) |
| 生成时间 | 2026-11-25T07:53:43+0000 |
| host / git | `node-61-47-19-68` / `e246e9fa3e` |
| 事件数 / 带栈事件数 | 1141811 / 149967 (13.1%) |
| 目标清单 | --ops-re（7 条） |
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
| `BatchMatMul` | 400 | 240 | 22.9 | 3 | `.../models/deepseek_v4.py:619  _apply_wo_a_bf16_matmul` |
| `Matmul` | 400 | 400 | 98.1 | 19 | `.../quantization/linear_method_npu.py:141  apply` |
| `QuantMatmul` | 400 | 256 | 27.8 | 6 | `.../quantization/linear_method_npu.py:141  apply` |
| `InplaceCopy` | 400 | 400 | 33.2 | 78 | `.../distributed/parallel_state.py:1433  all_gather` |
| `ReduceSum` | 400 | 400 | 29.9 | 12 | `.../dsv4/dsv4_mhc.py:34  mix_stats_torch` |
| `Div` | 400 | 400 | 61.3 | 16 | `.../dsv4/dsv4_mhc.py:34  mix_stats_torch` |
| `Embedding` | 400 | 4 | 220.1 | 1 | `.../quantization/unquant.py:954  embedding` |

## 逐算子

### `BatchMatMul` — 命中 400 / 带栈 240，合计 22.9 ms，3 种栈

**栈 #1 ×80**  [cpu_op]  dur 0.1 / 0.1 / 0.2 ms (min/med/max)  launcher: `.../models/deepseek_v4.py:619  _apply_wo_a_bf16_matmul`

仓库帧链：`.../models/deepseek_v4.py:619  _apply_wo_a_bf16_matmul ⟵ .../models/deepseek_v4.py:3185  _project_wo_a ⟵ .../models/deepseek_v4.py:3023  forward ⟵ .../models/deepseek_v4_platform_stage.py:162  run_attn_hc ⟵ …(共 25 个仓库帧)`

```
← .../torch/functional.py:372  einsum
← .../models/deepseek_v4.py:619  _apply_wo_a_bf16_matmul
← .../models/deepseek_v4.py:3185  _project_wo_a
← .../models/deepseek_v4.py:3023  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4_platform_stage.py:162  run_attn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← …（共 36 帧，省略最外层 22 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #2 ×80**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../dsv4/dsv4_mhc.py:47  post_torch`

仓库帧链：`.../dsv4/dsv4_mhc.py:47  post_torch ⟵ .../models/deepseek_v4_mhc.py:691  post ⟵ .../models/deepseek_v4_mhc.py:935  _post_fusion ⟵ .../models/deepseek_v4_mhc.py:1053  run_moe_post ⟵ …(共 26 个仓库帧)`

```
← .../dsv4/dsv4_mhc.py:47  post_torch
← .../models/deepseek_v4_mhc.py:691  post
← .../models/deepseek_v4_mhc.py:935  _post_fusion
← .../models/deepseek_v4_mhc.py:1053  run_moe_post
← .../models/deepseek_v4_platform_stage.py:199  run_ffn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← .../runner/eager_runner.py:339  _execute_extend
← .../runner/eager_runner.py:217  execute
← …（共 34 帧，省略最外层 20 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #3 ×80**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../dsv4/dsv4_mhc.py:47  post_torch`

仓库帧链：`.../dsv4/dsv4_mhc.py:47  post_torch ⟵ .../models/deepseek_v4_mhc.py:691  post ⟵ .../models/deepseek_v4_mhc.py:935  _post_fusion ⟵ .../models/deepseek_v4_mhc.py:970  run_attn_post ⟵ …(共 26 个仓库帧)`

```
← .../dsv4/dsv4_mhc.py:47  post_torch
← .../models/deepseek_v4_mhc.py:691  post
← .../models/deepseek_v4_mhc.py:935  _post_fusion
← .../models/deepseek_v4_mhc.py:970  run_attn_post
← .../models/deepseek_v4_platform_stage.py:170  run_attn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← .../runner/eager_runner.py:339  _execute_extend
← .../runner/eager_runner.py:217  execute
← …（共 34 帧，省略最外层 20 帧）
方向: 未确定（按原始顺序打印）
```

### `Matmul` — 命中 400 / 带栈 400，合计 98.1 ms，19 种栈

**栈 #1 ×30**  [cpu_op]  dur 0.1 / 0.4 / 0.9 ms (min/med/max)  launcher: `.../quantization/linear_method_npu.py:141  apply`

仓库帧链：`.../quantization/linear_method_npu.py:141  apply ⟵ .../schemes/modelslim_w8a8_int8.py:125  apply_weights ⟵ .../modelslim/modelslim.py:1076  apply ⟵ .../layers/linear.py:632  forward ⟵ …(共 31 个仓库帧)`

```
← .../torch/_ops.py:1279  __call__
← .../quantization/linear_method_npu.py:141  apply
← .../schemes/modelslim_w8a8_int8.py:125  apply_weights
← .../modelslim/modelslim.py:1076  apply
← .../layers/linear.py:632  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v2.py:401  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v2.py:1811  _forward_shared_experts
← .../models/deepseek_v2.py:1611  forward_deepep
← .../models/deepseek_v2.py:1170  forward
← .../modules/module.py:1789  _call_impl
← …（共 46 帧，省略最外层 32 帧）
方向: inner-first（本来就是最内层在前）
```

**栈 #2 ×30**  [cpu_op]  dur 0.1 / 0.4 / 0.9 ms (min/med/max)  launcher: `.../quantization/linear_method_npu.py:141  apply`

仓库帧链：`.../quantization/linear_method_npu.py:141  apply ⟵ .../schemes/modelslim_w8a8_int8.py:125  apply_weights ⟵ .../modelslim/modelslim.py:1076  apply ⟵ .../layers/linear.py:1810  forward ⟵ …(共 31 个仓库帧)`

```
← .../torch/_ops.py:1279  __call__
← .../quantization/linear_method_npu.py:141  apply
← .../schemes/modelslim_w8a8_int8.py:125  apply_weights
← .../modelslim/modelslim.py:1076  apply
← .../layers/linear.py:1810  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v2.py:501  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v2.py:1811  _forward_shared_experts
← .../models/deepseek_v2.py:1611  forward_deepep
← .../models/deepseek_v2.py:1170  forward
← .../modules/module.py:1789  _call_impl
← …（共 46 帧，省略最外层 32 帧）
方向: inner-first（本来就是最内层在前）
```

**栈 #3 ×30**  [cpu_op]  dur 0.1 / 0.3 / 0.8 ms (min/med/max)  launcher: `.../moe/matmul.py:41  forward`

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
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v2.py:1772  forward_deepep
← …（共 48 帧，省略最外层 34 帧）
方向: inner-first（本来就是最内层在前）
```

**栈 #4 ×30**  [cpu_op]  dur 0.1 / 0.3 / 0.8 ms (min/med/max)  launcher: `.../moe/matmul.py:41  forward`

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
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v2.py:1772  forward_deepep
← …（共 48 帧，省略最外层 34 帧）
方向: inner-first（本来就是最内层在前）
```

**栈 #5 ×30**  [cpu_op]  dur 0.1 / 0.3 / 0.9 ms (min/med/max)  launcher: `.../quantization/linear_method_npu.py:141  apply`

仓库帧链：`.../quantization/linear_method_npu.py:141  apply ⟵ .../schemes/modelslim_w8a8_int8.py:125  apply_weights ⟵ .../modelslim/modelslim.py:1076  apply ⟵ .../layers/linear.py:632  forward ⟵ …(共 28 个仓库帧)`

```
← .../torch/_ops.py:1279  __call__
← .../quantization/linear_method_npu.py:141  apply
← .../schemes/modelslim_w8a8_int8.py:125  apply_weights
← .../modelslim/modelslim.py:1076  apply
← .../layers/linear.py:632  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4.py:2506  _forward_prepare
← .../models/deepseek_v4.py:2910  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4_platform_stage.py:162  run_attn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← …（共 41 帧，省略最外层 27 帧）
方向: inner-first（本来就是最内层在前）
```

**栈 #6 ×30**  [cpu_op]  dur 0.1 / 0.2 / 0.4 ms (min/med/max)  launcher: `.../models/deepseek_v2.py:595  forward`

仓库帧链：`.../models/deepseek_v2.py:595  forward ⟵ .../models/deepseek_v2.py:1595  forward_deepep ⟵ .../models/deepseek_v2.py:1170  forward ⟵ .../models/deepseek_v4_platform_stage.py:352  _run_moe_ffn_dp_sync ⟵ …(共 26 个仓库帧)`

```
← .../models/deepseek_v2.py:595  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v2.py:1595  forward_deepep
← .../models/deepseek_v2.py:1170  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4_platform_stage.py:352  _run_moe_ffn_dp_sync
← .../models/deepseek_v4_platform_stage.py:191  run_ffn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← …（共 38 帧，省略最外层 24 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #7 ×30**  [cpu_op]  dur 0.1 / 0.2 / 0.4 ms (min/med/max)  launcher: `.../quantization/unquant.py:1088  apply`

仓库帧链：`.../quantization/unquant.py:1088  apply ⟵ .../layers/linear.py:390  forward ⟵ .../models/deepseek_v4.py:2313  _forward_prepare ⟵ .../models/deepseek_v4.py:2910  forward ⟵ …(共 26 个仓库帧)`

```
← .../quantization/unquant.py:1088  apply
← .../layers/linear.py:390  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4.py:2313  _forward_prepare
← .../models/deepseek_v4.py:2910  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4_platform_stage.py:162  run_attn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← …（共 38 帧，省略最外层 24 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #8 ×30**  [cpu_op]  dur 0.1 / 0.2 / 0.4 ms (min/med/max)  launcher: `.../quantization/unquant.py:1088  apply`

仓库帧链：`.../quantization/unquant.py:1088  apply ⟵ .../layers/linear.py:1810  forward ⟵ .../models/deepseek_v4.py:3222  _project_wo_b ⟵ .../models/deepseek_v4.py:3030  forward ⟵ …(共 26 个仓库帧)`

```
← .../quantization/unquant.py:1088  apply
← .../layers/linear.py:1810  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4.py:3222  _project_wo_b
← .../models/deepseek_v4.py:3030  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4_platform_stage.py:162  run_attn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← …（共 38 帧，省略最外层 24 帧）
方向: 未确定（按原始顺序打印）
```

_（另有 11 种较少见的栈未打印，`--stacks` 可调）_

### `QuantMatmul` — 命中 400 / 带栈 256，合计 27.8 ms，6 种栈

**栈 #1 ×80**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../quantization/linear_method_npu.py:141  apply`

仓库帧链：`.../quantization/linear_method_npu.py:141  apply ⟵ .../schemes/modelslim_w8a8_int8.py:125  apply_weights ⟵ .../modelslim/modelslim.py:1076  apply ⟵ .../layers/linear.py:632  forward ⟵ …(共 31 个仓库帧)`

```
← .../torch/_ops.py:1279  __call__
← .../quantization/linear_method_npu.py:141  apply
← .../schemes/modelslim_w8a8_int8.py:125  apply_weights
← .../modelslim/modelslim.py:1076  apply
← .../layers/linear.py:632  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v2.py:401  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v2.py:1811  _forward_shared_experts
← .../models/deepseek_v2.py:1611  forward_deepep
← .../models/deepseek_v2.py:1170  forward
← .../modules/module.py:1789  _call_impl
← …（共 46 帧，省略最外层 32 帧）
方向: inner-first（本来就是最内层在前）
```

**栈 #2 ×80**  [cpu_op]  dur 0.1 / 0.1 / 0.2 ms (min/med/max)  launcher: `.../quantization/linear_method_npu.py:141  apply`

仓库帧链：`.../quantization/linear_method_npu.py:141  apply ⟵ .../schemes/modelslim_w8a8_int8.py:125  apply_weights ⟵ .../modelslim/modelslim.py:1076  apply ⟵ .../layers/linear.py:1810  forward ⟵ …(共 31 个仓库帧)`

```
← .../torch/_ops.py:1279  __call__
← .../quantization/linear_method_npu.py:141  apply
← .../schemes/modelslim_w8a8_int8.py:125  apply_weights
← .../modelslim/modelslim.py:1076  apply
← .../layers/linear.py:1810  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v2.py:501  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v2.py:1811  _forward_shared_experts
← .../models/deepseek_v2.py:1611  forward_deepep
← .../models/deepseek_v2.py:1170  forward
← .../modules/module.py:1789  _call_impl
← …（共 46 帧，省略最外层 32 帧）
方向: inner-first（本来就是最内层在前）
```

**栈 #3 ×80**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../quantization/linear_method_npu.py:141  apply`

仓库帧链：`.../quantization/linear_method_npu.py:141  apply ⟵ .../schemes/modelslim_w8a8_int8.py:125  apply_weights ⟵ .../modelslim/modelslim.py:1076  apply ⟵ .../layers/linear.py:632  forward ⟵ …(共 28 个仓库帧)`

```
← .../torch/_ops.py:1279  __call__
← .../quantization/linear_method_npu.py:141  apply
← .../schemes/modelslim_w8a8_int8.py:125  apply_weights
← .../modelslim/modelslim.py:1076  apply
← .../layers/linear.py:632  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4.py:2506  _forward_prepare
← .../models/deepseek_v4.py:2910  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4_platform_stage.py:162  run_attn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← …（共 41 帧，省略最外层 27 帧）
方向: inner-first（本来就是最内层在前）
```

**栈 #4 ×8**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../quantization/linear_method_npu.py:141  apply`

仓库帧链：`.../quantization/linear_method_npu.py:141  apply ⟵ .../schemes/modelslim_w8a8_int8.py:125  apply_weights ⟵ .../modelslim/modelslim.py:1076  apply ⟵ .../layers/linear.py:390  forward ⟵ …(共 32 个仓库帧)`

```
← .../torch/_ops.py:1279  __call__
← .../quantization/linear_method_npu.py:141  apply
← .../schemes/modelslim_w8a8_int8.py:125  apply_weights
← .../modelslim/modelslim.py:1076  apply
← .../layers/linear.py:390  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../attention/ascend_dsv4_backend.py:1544  _dsv41_index_q
← .../attention/ascend_dsv4_backend.py:1753  _dsv41_index_qsli_topk
← .../attention/ascend_dsv4_backend.py:1443  _low_ratio_index_topk
← .../attention/ascend_dsv4_backend.py:1338  forward_low_ratio_sources
← .../models/deepseek_v4.py:2626  _forward_prepare
← .../models/deepseek_v4.py:2910  forward
← .../modules/module.py:1789  _call_impl
← …（共 45 帧，省略最外层 31 帧）
方向: inner-first（本来就是最内层在前）
```

**栈 #5 ×6**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../quantization/linear_method_npu.py:141  apply`

仓库帧链：`.../quantization/linear_method_npu.py:141  apply ⟵ .../schemes/modelslim_w8a8_int8.py:125  apply_weights ⟵ .../modelslim/modelslim.py:1076  apply ⟵ .../layers/linear.py:390  forward ⟵ …(共 32 个仓库帧)`

```
← .../torch/_ops.py:1279  __call__
← .../quantization/linear_method_npu.py:141  apply
← .../schemes/modelslim_w8a8_int8.py:125  apply_weights
← .../modelslim/modelslim.py:1076  apply
← .../layers/linear.py:390  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../attention/ascend_dsv4_backend.py:1544  _dsv41_index_q
← .../attention/ascend_dsv4_backend.py:1593  _dsv41_index_full_topk
← .../attention/ascend_dsv4_backend.py:1445  _low_ratio_index_topk
← .../attention/ascend_dsv4_backend.py:1338  forward_low_ratio_sources
← .../models/deepseek_v4.py:2626  _forward_prepare
← .../models/deepseek_v4.py:2910  forward
← .../modules/module.py:1789  _call_impl
← …（共 45 帧，省略最外层 31 帧）
方向: inner-first（本来就是最内层在前）
```

**栈 #6 ×2**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../quantization/linear_method_npu.py:141  apply`

仓库帧链：`.../quantization/linear_method_npu.py:141  apply ⟵ .../schemes/modelslim_w8a8_int8.py:125  apply_weights ⟵ .../modelslim/modelslim.py:1076  apply ⟵ .../layers/linear.py:390  forward ⟵ …(共 32 个仓库帧)`

```
← .../torch/_ops.py:1279  __call__
← .../quantization/linear_method_npu.py:141  apply
← .../schemes/modelslim_w8a8_int8.py:125  apply_weights
← .../modelslim/modelslim.py:1076  apply
← .../layers/linear.py:390  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../attention/ascend_dsv4_backend.py:1544  _dsv41_index_q
← .../attention/ascend_dsv4_backend.py:1593  _dsv41_index_full_topk
← .../attention/ascend_dsv4_backend.py:1438  _low_ratio_index_topk
← .../attention/ascend_dsv4_backend.py:1338  forward_low_ratio_sources
← .../models/deepseek_v4.py:2626  _forward_prepare
← .../models/deepseek_v4.py:2910  forward
← .../modules/module.py:1789  _call_impl
← …（共 45 帧，省略最外层 31 帧）
方向: inner-first（本来就是最内层在前）
```

### `InplaceCopy` — 命中 400 / 带栈 400，合计 33.2 ms，78 种栈

**栈 #1 ×48**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../distributed/parallel_state.py:1433  all_gather`

仓库帧链：`.../distributed/parallel_state.py:1433  all_gather ⟵ .../layers/dp_attention.py:1299  attn_tp_all_gather ⟵ .../deepseek_common/moe_transport.py:290  return_output ⟵ .../models/deepseek_v4_platform_stage.py:361  _run_moe_ffn_dp_sync ⟵ …(共 26 个仓库帧)`

```
← .../distributed/distributed_c10d.py:4280  all_gather
← .../distributed/c10d_logger.py:83  wrapper
← .../distributed/parallel_state.py:1433  all_gather
← .../layers/dp_attention.py:1299  attn_tp_all_gather
← .../deepseek_common/moe_transport.py:290  return_output
← .../models/deepseek_v4_platform_stage.py:361  _run_moe_ffn_dp_sync
← .../models/deepseek_v4_platform_stage.py:191  run_ffn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← …（共 36 帧，省略最外层 22 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #2 ×24**  [cpu_op]  dur 0.1 / 0.1 / 0.2 ms (min/med/max)  launcher: `.../token_dispatcher/deepep.py:668  _dispatch_core`

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
← .../ep_moe/layer.py:245  forward_impl
← .../ep_moe/layer.py:236  forward
← …（共 51 帧，省略最外层 37 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #3 ×12**  [cpu_op]  dur 0.1 / 0.1 / 0.2 ms (min/med/max)  launcher: `.../moe/activation.py:72  _clipped_swiglu_quant`

仓库帧链：`.../moe/activation.py:72  _clipped_swiglu_quant ⟵ .../moe/activation.py:152  _apply_activation ⟵ .../moe_runner/ascend.py:299  run ⟵ .../moe_runner/runner.py:254  run ⟵ …(共 35 个仓库帧)`

```
← .../contrib/transfer_to_npu.py:194  decorated
← .../moe/activation.py:72  _clipped_swiglu_quant
← .../moe/activation.py:152  _apply_activation
← .../moe_runner/ascend.py:299  run
← .../moe_runner/runner.py:254  run
← .../modelslim/modelslim.py:1157  apply
← .../fused_moe_triton/layer.py:1837  run_moe_core
← .../ep_moe/layer.py:272  run_moe_core
← .../fused_moe_triton/layer.py:1772  forward_impl
← .../ep_moe/layer.py:245  forward_impl
← .../ep_moe/layer.py:236  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v2.py:1772  forward_deepep
← …（共 48 帧，省略最外层 34 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #4 ×12**  [cpu_op]  dur 0.1 / 0.1 / 0.2 ms (min/med/max)  launcher: `.../token_dispatcher/deepep.py:723  _combine_core`

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
← .../ep_moe/layer.py:236  forward
← .../modules/module.py:1789  _call_impl
← …（共 50 帧，省略最外层 36 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #5 ×12**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../moe/activation.py:66  _clipped_swiglu_quant`

仓库帧链：`.../moe/activation.py:66  _clipped_swiglu_quant ⟵ .../moe/activation.py:152  _apply_activation ⟵ .../moe_runner/ascend.py:299  run ⟵ .../moe_runner/runner.py:254  run ⟵ …(共 35 个仓库帧)`

```
← .../moe/activation.py:66  _clipped_swiglu_quant
← .../moe/activation.py:152  _apply_activation
← .../moe_runner/ascend.py:299  run
← .../moe_runner/runner.py:254  run
← .../modelslim/modelslim.py:1157  apply
← .../fused_moe_triton/layer.py:1837  run_moe_core
← .../ep_moe/layer.py:272  run_moe_core
← .../fused_moe_triton/layer.py:1772  forward_impl
← .../ep_moe/layer.py:245  forward_impl
← .../ep_moe/layer.py:236  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v2.py:1772  forward_deepep
← .../models/deepseek_v2.py:1170  forward
← …（共 47 帧，省略最外层 33 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #6 ×12**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../moe/activation.py:67  _clipped_swiglu_quant`

仓库帧链：`.../moe/activation.py:67  _clipped_swiglu_quant ⟵ .../moe/activation.py:152  _apply_activation ⟵ .../moe_runner/ascend.py:299  run ⟵ .../moe_runner/runner.py:254  run ⟵ …(共 35 个仓库帧)`

```
← .../moe/activation.py:67  _clipped_swiglu_quant
← .../moe/activation.py:152  _apply_activation
← .../moe_runner/ascend.py:299  run
← .../moe_runner/runner.py:254  run
← .../modelslim/modelslim.py:1157  apply
← .../fused_moe_triton/layer.py:1837  run_moe_core
← .../ep_moe/layer.py:272  run_moe_core
← .../fused_moe_triton/layer.py:1772  forward_impl
← .../ep_moe/layer.py:245  forward_impl
← .../ep_moe/layer.py:236  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v2.py:1772  forward_deepep
← .../models/deepseek_v2.py:1170  forward
← …（共 47 帧，省略最外层 33 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #7 ×12**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../token_dispatcher/deepep.py:564  dispatch_a`

仓库帧链：`.../token_dispatcher/deepep.py:564  dispatch_a ⟵ .../token_dispatcher/deepep.py:1064  dispatch_a ⟵ .../token_dispatcher/deepep.py:1052  dispatch ⟵ .../batch_overlap/two_batch_overlap.py:1193  _execute ⟵ …(共 34 个仓库帧)`

```
← .../contrib/transfer_to_npu.py:194  decorated
← .../token_dispatcher/deepep.py:564  dispatch_a
← .../token_dispatcher/deepep.py:1064  dispatch_a
← .../token_dispatcher/deepep.py:1052  dispatch
← .../batch_overlap/two_batch_overlap.py:1193  _execute
← .../batch_overlap/two_batch_overlap.py:1196  dispatch
← .../fused_moe_triton/layer.py:1800  _dispatch_with_pre_quant
← .../fused_moe_triton/layer.py:1768  forward_impl
← .../ep_moe/layer.py:245  forward_impl
← .../ep_moe/layer.py:236  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v2.py:1772  forward_deepep
← .../models/deepseek_v2.py:1170  forward
← …（共 47 帧，省略最外层 33 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #8 ×12**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../attention/ascend_dsv4_backend.py:2148  _dsv41_a3_kv_lengths`

仓库帧链：`.../attention/ascend_dsv4_backend.py:2148  _dsv41_a3_kv_lengths ⟵ .../attention/ascend_dsv4_backend.py:2153  _dsv41_a3_attention_kwargs ⟵ .../attention/ascend_dsv4_backend.py:2204  _forward_low_ratio_attention ⟵ .../attention/ascend_dsv4_backend.py:4698  forward ⟵ …(共 27 个仓库帧)`

```
← .../contrib/transfer_to_npu.py:194  decorated
← .../attention/ascend_dsv4_backend.py:2148  _dsv41_a3_kv_lengths
← .../attention/ascend_dsv4_backend.py:2153  _dsv41_a3_attention_kwargs
← .../attention/ascend_dsv4_backend.py:2204  _forward_low_ratio_attention
← .../attention/ascend_dsv4_backend.py:4698  forward
← .../models/deepseek_v4.py:3010  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4_platform_stage.py:162  run_attn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← …（共 38 帧，省略最外层 24 帧）
方向: 未确定（按原始顺序打印）
```

_（另有 70 种较少见的栈未打印，`--stacks` 可调）_

### `ReduceSum` — 命中 400 / 带栈 400，合计 29.9 ms，12 种栈

**栈 #1 ×95**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../dsv4/dsv4_mhc.py:34  mix_stats_torch`

仓库帧链：`.../dsv4/dsv4_mhc.py:34  mix_stats_torch ⟵ .../models/deepseek_v4_mhc.py:583  _npu_combine_premix ⟵ .../models/deepseek_v4_mhc.py:619  _combine ⟵ .../models/deepseek_v4_mhc.py:549  combine ⟵ …(共 26 个仓库帧)`

```
← .../dsv4/dsv4_mhc.py:34  mix_stats_torch
← .../models/deepseek_v4_mhc.py:583  _npu_combine_premix
← .../models/deepseek_v4_mhc.py:619  _combine
← .../models/deepseek_v4_mhc.py:549  combine
← .../models/deepseek_v4_platform_stage.py:153  run_attn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← .../runner/eager_runner.py:339  _execute_extend
← .../runner/eager_runner.py:217  execute
← …（共 34 帧，省略最外层 20 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #2 ×95**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../dsv4/dsv4_mhc.py:35  mix_stats_torch`

仓库帧链：`.../dsv4/dsv4_mhc.py:35  mix_stats_torch ⟵ .../models/deepseek_v4_mhc.py:583  _npu_combine_premix ⟵ .../models/deepseek_v4_mhc.py:619  _combine ⟵ .../models/deepseek_v4_mhc.py:549  combine ⟵ …(共 26 个仓库帧)`

```
← .../dsv4/dsv4_mhc.py:35  mix_stats_torch
← .../models/deepseek_v4_mhc.py:583  _npu_combine_premix
← .../models/deepseek_v4_mhc.py:619  _combine
← .../models/deepseek_v4_mhc.py:549  combine
← .../models/deepseek_v4_platform_stage.py:153  run_attn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← .../runner/eager_runner.py:339  _execute_extend
← .../runner/eager_runner.py:217  execute
← …（共 34 帧，省略最外层 20 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #3 ×91**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../dsv4/dsv4_mhc.py:34  mix_stats_torch`

仓库帧链：`.../dsv4/dsv4_mhc.py:34  mix_stats_torch ⟵ .../models/deepseek_v4_mhc.py:583  _npu_combine_premix ⟵ .../models/deepseek_v4_mhc.py:619  _combine ⟵ .../models/deepseek_v4_mhc.py:549  combine ⟵ …(共 26 个仓库帧)`

```
← .../dsv4/dsv4_mhc.py:34  mix_stats_torch
← .../models/deepseek_v4_mhc.py:583  _npu_combine_premix
← .../models/deepseek_v4_mhc.py:619  _combine
← .../models/deepseek_v4_mhc.py:549  combine
← .../models/deepseek_v4_platform_stage.py:184  run_ffn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← .../runner/eager_runner.py:339  _execute_extend
← .../runner/eager_runner.py:217  execute
← …（共 34 帧，省略最外层 20 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #4 ×91**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../dsv4/dsv4_mhc.py:35  mix_stats_torch`

仓库帧链：`.../dsv4/dsv4_mhc.py:35  mix_stats_torch ⟵ .../models/deepseek_v4_mhc.py:583  _npu_combine_premix ⟵ .../models/deepseek_v4_mhc.py:619  _combine ⟵ .../models/deepseek_v4_mhc.py:549  combine ⟵ …(共 26 个仓库帧)`

```
← .../dsv4/dsv4_mhc.py:35  mix_stats_torch
← .../models/deepseek_v4_mhc.py:583  _npu_combine_premix
← .../models/deepseek_v4_mhc.py:619  _combine
← .../models/deepseek_v4_mhc.py:549  combine
← .../models/deepseek_v4_platform_stage.py:184  run_ffn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← .../runner/eager_runner.py:339  _execute_extend
← .../runner/eager_runner.py:217  execute
← …（共 34 帧，省略最外层 20 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #5 ×5**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../dsv4/dsv4_mhc.py:32  mix_stats_torch`

仓库帧链：`.../dsv4/dsv4_mhc.py:32  mix_stats_torch ⟵ .../models/deepseek_v4_mhc.py:583  _npu_combine_premix ⟵ .../models/deepseek_v4_mhc.py:619  _combine ⟵ .../models/deepseek_v4_mhc.py:549  combine ⟵ …(共 26 个仓库帧)`

```
← .../dsv4/dsv4_mhc.py:32  mix_stats_torch
← .../models/deepseek_v4_mhc.py:583  _npu_combine_premix
← .../models/deepseek_v4_mhc.py:619  _combine
← .../models/deepseek_v4_mhc.py:549  combine
← .../models/deepseek_v4_platform_stage.py:184  run_ffn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← .../runner/eager_runner.py:339  _execute_extend
← .../runner/eager_runner.py:217  execute
← …（共 34 帧，省略最外层 20 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #6 ×5**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../dsv4/dsv4_mhc.py:32  mix_stats_torch`

仓库帧链：`.../dsv4/dsv4_mhc.py:32  mix_stats_torch ⟵ .../models/deepseek_v4_mhc.py:583  _npu_combine_premix ⟵ .../models/deepseek_v4_mhc.py:619  _combine ⟵ .../models/deepseek_v4_mhc.py:549  combine ⟵ …(共 26 个仓库帧)`

```
← .../dsv4/dsv4_mhc.py:32  mix_stats_torch
← .../models/deepseek_v4_mhc.py:583  _npu_combine_premix
← .../models/deepseek_v4_mhc.py:619  _combine
← .../models/deepseek_v4_mhc.py:549  combine
← .../models/deepseek_v4_platform_stage.py:153  run_attn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← .../runner/eager_runner.py:339  _execute_extend
← .../runner/eager_runner.py:217  execute
← …（共 34 帧，省略最外层 20 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #7 ×4**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../models/deepseek_v4_mhc.py:672  combine_premix`

仓库帧链：`.../models/deepseek_v4_mhc.py:672  combine_premix ⟵ .../models/deepseek_v4_mhc.py:584  _npu_combine_premix ⟵ .../models/deepseek_v4_mhc.py:619  _combine ⟵ .../models/deepseek_v4_mhc.py:549  combine ⟵ …(共 26 个仓库帧)`

```
← .../models/deepseek_v4_mhc.py:672  combine_premix
← .../models/deepseek_v4_mhc.py:584  _npu_combine_premix
← .../models/deepseek_v4_mhc.py:619  _combine
← .../models/deepseek_v4_mhc.py:549  combine
← .../models/deepseek_v4_platform_stage.py:184  run_ffn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← .../runner/eager_runner.py:339  _execute_extend
← .../runner/eager_runner.py:217  execute
← …（共 34 帧，省略最外层 20 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #8 ×4**  [cpu_op]  dur 0.1 / 0.1 / 0.1 ms (min/med/max)  launcher: `.../token_dispatcher/deepep.py:668  _dispatch_core`

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
← .../ep_moe/layer.py:245  forward_impl
← .../ep_moe/layer.py:236  forward
← …（共 51 帧，省略最外层 37 帧）
方向: 未确定（按原始顺序打印）
```

_（另有 4 种较少见的栈未打印，`--stacks` 可调）_

### `Div` — 命中 400 / 带栈 400，合计 61.3 ms，16 种栈

**栈 #1 ×106**  [cpu_op]  dur 0.1 / 0.2 / 0.4 ms (min/med/max)  launcher: `.../dsv4/dsv4_mhc.py:34  mix_stats_torch`

仓库帧链：`.../dsv4/dsv4_mhc.py:34  mix_stats_torch ⟵ .../models/deepseek_v4_mhc.py:583  _npu_combine_premix ⟵ .../models/deepseek_v4_mhc.py:619  _combine ⟵ .../models/deepseek_v4_mhc.py:549  combine ⟵ …(共 26 个仓库帧)`

```
← .../dsv4/dsv4_mhc.py:34  mix_stats_torch
← .../models/deepseek_v4_mhc.py:583  _npu_combine_premix
← .../models/deepseek_v4_mhc.py:619  _combine
← .../models/deepseek_v4_mhc.py:549  combine
← .../models/deepseek_v4_platform_stage.py:153  run_attn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← .../runner/eager_runner.py:339  _execute_extend
← .../runner/eager_runner.py:217  execute
← …（共 34 帧，省略最外层 20 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #2 ×106**  [cpu_op]  dur 0.1 / 0.1 / 0.3 ms (min/med/max)  launcher: `.../dsv4/dsv4_mhc.py:35  mix_stats_torch`

仓库帧链：`.../dsv4/dsv4_mhc.py:35  mix_stats_torch ⟵ .../models/deepseek_v4_mhc.py:583  _npu_combine_premix ⟵ .../models/deepseek_v4_mhc.py:619  _combine ⟵ .../models/deepseek_v4_mhc.py:549  combine ⟵ …(共 26 个仓库帧)`

```
← .../dsv4/dsv4_mhc.py:35  mix_stats_torch
← .../models/deepseek_v4_mhc.py:583  _npu_combine_premix
← .../models/deepseek_v4_mhc.py:619  _combine
← .../models/deepseek_v4_mhc.py:549  combine
← .../models/deepseek_v4_platform_stage.py:153  run_attn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← .../runner/eager_runner.py:339  _execute_extend
← .../runner/eager_runner.py:217  execute
← …（共 34 帧，省略最外层 20 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #3 ×76**  [cpu_op]  dur 0.1 / 0.2 / 0.3 ms (min/med/max)  launcher: `.../dsv4/dsv4_mhc.py:34  mix_stats_torch`

仓库帧链：`.../dsv4/dsv4_mhc.py:34  mix_stats_torch ⟵ .../models/deepseek_v4_mhc.py:583  _npu_combine_premix ⟵ .../models/deepseek_v4_mhc.py:619  _combine ⟵ .../models/deepseek_v4_mhc.py:549  combine ⟵ …(共 26 个仓库帧)`

```
← .../dsv4/dsv4_mhc.py:34  mix_stats_torch
← .../models/deepseek_v4_mhc.py:583  _npu_combine_premix
← .../models/deepseek_v4_mhc.py:619  _combine
← .../models/deepseek_v4_mhc.py:549  combine
← .../models/deepseek_v4_platform_stage.py:184  run_ffn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← .../runner/eager_runner.py:339  _execute_extend
← .../runner/eager_runner.py:217  execute
← …（共 34 帧，省略最外层 20 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #4 ×76**  [cpu_op]  dur 0.1 / 0.1 / 0.3 ms (min/med/max)  launcher: `.../dsv4/dsv4_mhc.py:35  mix_stats_torch`

仓库帧链：`.../dsv4/dsv4_mhc.py:35  mix_stats_torch ⟵ .../models/deepseek_v4_mhc.py:583  _npu_combine_premix ⟵ .../models/deepseek_v4_mhc.py:619  _combine ⟵ .../models/deepseek_v4_mhc.py:549  combine ⟵ …(共 26 个仓库帧)`

```
← .../dsv4/dsv4_mhc.py:35  mix_stats_torch
← .../models/deepseek_v4_mhc.py:583  _npu_combine_premix
← .../models/deepseek_v4_mhc.py:619  _combine
← .../models/deepseek_v4_mhc.py:549  combine
← .../models/deepseek_v4_platform_stage.py:184  run_ffn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← .../runner/eager_runner.py:339  _execute_extend
← .../runner/eager_runner.py:217  execute
← …（共 34 帧，省略最外层 20 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #5 ×6**  [cpu_op]  dur 0.1 / 0.2 / 0.3 ms (min/med/max)  launcher: `.../dsv4/dsv4_mhc.py:32  mix_stats_torch`

仓库帧链：`.../dsv4/dsv4_mhc.py:32  mix_stats_torch ⟵ .../models/deepseek_v4_mhc.py:583  _npu_combine_premix ⟵ .../models/deepseek_v4_mhc.py:619  _combine ⟵ .../models/deepseek_v4_mhc.py:549  combine ⟵ …(共 26 个仓库帧)`

```
← .../dsv4/dsv4_mhc.py:32  mix_stats_torch
← .../models/deepseek_v4_mhc.py:583  _npu_combine_premix
← .../models/deepseek_v4_mhc.py:619  _combine
← .../models/deepseek_v4_mhc.py:549  combine
← .../models/deepseek_v4_platform_stage.py:153  run_attn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← .../runner/eager_runner.py:339  _execute_extend
← .../runner/eager_runner.py:217  execute
← …（共 34 帧，省略最外层 20 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #6 ×4**  [cpu_op]  dur 0.1 / 0.3 / 0.5 ms (min/med/max)  launcher: `.../moe/activation.py:69  _clipped_swiglu_quant`

仓库帧链：`.../moe/activation.py:69  _clipped_swiglu_quant ⟵ .../moe/activation.py:152  _apply_activation ⟵ .../moe_runner/ascend.py:299  run ⟵ .../moe_runner/runner.py:254  run ⟵ …(共 35 个仓库帧)`

```
← .../moe/activation.py:69  _clipped_swiglu_quant
← .../moe/activation.py:152  _apply_activation
← .../moe_runner/ascend.py:299  run
← .../moe_runner/runner.py:254  run
← .../modelslim/modelslim.py:1157  apply
← .../fused_moe_triton/layer.py:1837  run_moe_core
← .../ep_moe/layer.py:272  run_moe_core
← .../fused_moe_triton/layer.py:1772  forward_impl
← .../ep_moe/layer.py:245  forward_impl
← .../ep_moe/layer.py:236  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v2.py:1772  forward_deepep
← .../models/deepseek_v2.py:1170  forward
← …（共 47 帧，省略最外层 33 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #7 ×4**  [cpu_op]  dur 0.1 / 0.2 / 0.3 ms (min/med/max)  launcher: `.../moe/activation.py:71  _clipped_swiglu_quant`

仓库帧链：`.../moe/activation.py:71  _clipped_swiglu_quant ⟵ .../moe/activation.py:152  _apply_activation ⟵ .../moe_runner/ascend.py:299  run ⟵ .../moe_runner/runner.py:254  run ⟵ …(共 35 个仓库帧)`

```
← .../moe/activation.py:71  _clipped_swiglu_quant
← .../moe/activation.py:152  _apply_activation
← .../moe_runner/ascend.py:299  run
← .../moe_runner/runner.py:254  run
← .../modelslim/modelslim.py:1157  apply
← .../fused_moe_triton/layer.py:1837  run_moe_core
← .../ep_moe/layer.py:272  run_moe_core
← .../fused_moe_triton/layer.py:1772  forward_impl
← .../ep_moe/layer.py:245  forward_impl
← .../ep_moe/layer.py:236  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v2.py:1772  forward_deepep
← .../models/deepseek_v2.py:1170  forward
← …（共 47 帧，省略最外层 33 帧）
方向: 未确定（按原始顺序打印）
```

**栈 #8 ×4**  [cpu_op]  dur 0.1 / 0.2 / 0.3 ms (min/med/max)  launcher: `.../dsv4/dsv4_mhc.py:32  mix_stats_torch`

仓库帧链：`.../dsv4/dsv4_mhc.py:32  mix_stats_torch ⟵ .../models/deepseek_v4_mhc.py:583  _npu_combine_premix ⟵ .../models/deepseek_v4_mhc.py:619  _combine ⟵ .../models/deepseek_v4_mhc.py:549  combine ⟵ …(共 26 个仓库帧)`

```
← .../dsv4/dsv4_mhc.py:32  mix_stats_torch
← .../models/deepseek_v4_mhc.py:583  _npu_combine_premix
← .../models/deepseek_v4_mhc.py:619  _combine
← .../models/deepseek_v4_mhc.py:549  combine
← .../models/deepseek_v4_platform_stage.py:184  run_ffn_hc
← .../models/deepseek_v4_platform_stage.py:209  forward_hc_pre_from_prev
← .../models/deepseek_v4.py:4430  _forward_layers_hc_pre_from_prev
← .../models/deepseek_v4.py:4702  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← .../runner/eager_runner.py:339  _execute_extend
← .../runner/eager_runner.py:217  execute
← …（共 34 帧，省略最外层 20 帧）
方向: 未确定（按原始顺序打印）
```

_（另有 8 种较少见的栈未打印，`--stacks` 可调）_

### `Embedding` — 命中 400 / 带栈 4，合计 220.1 ms，1 种栈

**栈 #1 ×4**  [cpu_op]  dur 0.1 / 0.1 / 0.3 ms (min/med/max)  launcher: `.../quantization/unquant.py:954  embedding`

仓库帧链：`.../quantization/unquant.py:954  embedding ⟵ .../layers/vocab_parallel_embedding.py:476  _embed_local_shard ⟵ .../layers/vocab_parallel_embedding.py:516  forward ⟵ .../models/deepseek_v4.py:4626  _forward_impl ⟵ …(共 22 个仓库帧)`

```
← .../nn/functional.py:2615  embedding
← .../quantization/unquant.py:954  embedding
← .../layers/vocab_parallel_embedding.py:476  _embed_local_shard
← .../layers/vocab_parallel_embedding.py:516  forward
← .../modules/module.py:1789  _call_impl
← .../modules/module.py:1778  _wrapped_call_impl
← .../models/deepseek_v4.py:4626  _forward_impl
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:4602  forward
← .../utils/_contextlib.py:124  decorate_context
← .../models/deepseek_v4.py:5173  forward
← .../runner/eager_runner.py:339  _execute_extend
← .../runner/eager_runner.py:217  execute
← .../model_executor/model_runner.py:2191  _forward_raw
← …（共 33 帧，省略最外层 19 帧）
方向: 未确定（按原始顺序打印）
```

## 回填用速查（可直接贴进 KERNEL_CODE_MAP.md 的映射表）

| 算子 | 本次 trace 次数 | launcher（Python 发起位置） |
|---|---|---|
| `BatchMatMul` | 400 | `.../models/deepseek_v4.py:619  _apply_wo_a_bf16_matmul` |
| `Matmul` | 400 | `.../quantization/linear_method_npu.py:141  apply` |
| `QuantMatmul` | 400 | `.../quantization/linear_method_npu.py:141  apply` |
| `InplaceCopy` | 400 | `.../distributed/parallel_state.py:1433  all_gather` |
| `ReduceSum` | 400 | `.../dsv4/dsv4_mhc.py:34  mix_stats_torch` |
| `Div` | 400 | `.../dsv4/dsv4_mhc.py:34  mix_stats_torch` |
| `Embedding` | 400 | `.../quantization/unquant.py:954  embedding` |
