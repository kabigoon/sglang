# 设计文档：将 Ascend `ascend_tp` 的 MoE dispatch 逻辑合并进社区 `none` 路径

| 项 | 内容 |
|---|---|
| 状态 | 已实现，已通过数值验收，**已合入目标分支**（`chenxu214/ci_sglang:feat/dev_ep`） |
| 分支 | `feat/moe-none-merge`（基于 `chenxu214/ci_sglang:feat/dev_ep` @ `d3c76503`） |
| 提交 | `3e17228`（单提交，15 文件，+292 / −230） |
| PR | `chenxu214/ci_sglang#8`（已合入） |
| 影响面 | Ascend NPU + `--moe-a2a-backend none`（默认值）；其它 backend 行为不变 |
| 参考 | 学习/调试笔记见 `DTS/moe_ascend_tp_learning_handoff.md` |

---

## 1. 需求介绍

### 1.1 背景

SGLang 的 MoE 分为两层选择：

| 层 | 参数 | 作用 |
|---|---|---|
| 调度层（通信/排布） | `--moe-a2a-backend` | 选择 token 的 dispatch/combine 实现（`none` / `deepep` / `ascend_ops` …） |
| 计算层（expert GEMM） | `--moe-runner-backend` | 选择 expert 计算实现（`triton` / `deep_gemm` / `ascend` …） |

Ascend NPU 上存在一个专属 dispatcher `AscendTPDispatcher`，由 `create_moe_dispatcher()` 在 `none + is_npu()` 时选中。它实际只做两件事：

1. `dispatch()`：调用 `npu_moe_init_routing_v2`，把 token 复制 top-k 份、按 expert 升序重排、顺带量化，产出 `(hs, expanded_row_idx, expert_tokens, scale)`；
2. `combine()`：调用 `npu_moe_finalize_routing`，把 `[T*K, H]` 散回 `[T, H]` 并做 top-k 加权求和。

**这两步没有任何通信语义**——`ascend_tp` 的含义是"expert 权重按 TP 切分，没有 EP all-to-all"。它占用了 dispatcher（通信/排布抽象）的位置，却只做本地数据重排，导致：

- NPU 的 `none` 分支与 GPU 的 `none` 分支走了两份完全不同的代码；
- 社区为"格式 × 算子族"准备的 permute 适配矩阵里，`("standard","ascend")` 这一格长期空缺；
- 任何跨 backend 的改动都要额外维护一条 Ascend 专用分支。

### 1.2 需求

把 `ascend_tp` 的 dispatch/combine 逻辑合并进社区 `none` 路径：

- NPU 的 `none` 分支改用社区 `StandardDispatcher`；
- Ascend 的"按 expert 重排 + 量化 + 反排布"下沉为 **MoE Runner 的 pre/post permute 钩子**（与 triton / deep_gemm / aiter / humming 的做法一致）；
- 删除 `ascend_tp` 相关的 dispatcher、格式枚举、CLI 取值；
- **行为逐位不变**。

### 1.3 验收标准

> 在不配置 `--moe-a2a-backend` 参数时，链路保持原有 `ascend_tp` 逻辑不变。

"不变"覆盖：token 排布（`expanded_row_idx`）、分组信息（`expert_tokens`）、量化结果（`hs` / `scale`）、expert 输出、combine 后的数值与顺序。

### 1.4 非目标

| 非目标 | 原因 |
|---|---|
| 不改 DeepEP / `ascend_fuseep` / `ascend_ops` 的 A2A 语义 | 它们是真正的 EP 通信路径，本次不涉及 |
| 不改 `AscendRunnerCore`（gmm1 / activation / gmm2） | 计算层与本次重构正交 |
| 不为 `none` 增加 EP 支持 | Ascend routing 按全局 expert 索引，与 EP 切分不兼容（见 §3.6） |
| 不改 GPU/ROCm 上 `none` 的行为 | 避免为 NPU 迁移污染通用语义 |
| 不为新增量化档位补"防御性断言" | 主干分支保持精简，遗留项见 §5.7 |

---

## 2. 功能和接口描述

### 2.1 用户可见变化

| 项 | 变更前 | 变更后 |
|---|---|---|
| `--moe-a2a-backend`（不传 / `none`） | 解析为 `none`，NPU 上选 `AscendTPDispatcher` | 解析为 `none`，NPU 上选 `StandardDispatcher` |
| `--moe-a2a-backend ascend_tp` | 运行时被归一化为 `none`（静默） | **参数解析期直接报错**（该取值已删除） |
| `--moe-a2a-backend none` + `--ep-size > 1`（NPU） | 运行时未定义行为（routing 与权重索引空间不一致） | **建层时显式报错**，提示改用 `deepep` / 保持 `ep_size=1` |
| 其它 backend | 不变 | 不变 |

### 2.2 删除的接口

| 符号 | 位置 | 说明 |
|---|---|---|
| `AscendTPDispatcher` / `AscendTPDispatchOutput` / `AscendTPCombineInput` | `layers/moe/token_dispatcher/ascend_tp.py`（整文件删除，143 行） | 其职责被 §2.3 的新接口接管 |
| `DispatchOutputFormat.ASCEND_TP`、`CombineInputFormat.ASCEND_TP` | `token_dispatcher/base.py` | 格式名随 dispatcher 一起消失 |
| `DispatchOutputChecker.format_is_ascend_tp`、`CombineInputChecker.format_is_ascend_tp` | 同上 | 无调用方 |
| `MoeA2ABackend.ASCEND_TP`、`is_ascend_tp()` | `layers/moe/utils.py` | 枚举值删除 |
| `ascend_tp` CLI 取值 | `arg_groups/fields/exec_.py`、`fields/spec.py` | 含 speculative（draft）那一处 |
| `ascend_tp → none` 归一化块 | `arg_groups/moe_hook.py` | 不可达，删除 |

### 2.3 新增/变更的接口

#### （a）`hardware_backend/npu/moe/dispatch.py`（新增 116 行）

routing 的**状态容器 + 内核工厂**。状态由量化方法在权重加载期写入，内核在首次前向按需构建并缓存。

| 函数 | 签名 | 语义 |
|---|---|---|
| `set_dispatch_dtype` | `(layer, dtype) -> None` | 记录 routing 应产出的激活 dtype（`bf16`/`int8`/`mxfp8`），写入 `layer.moe_dispatch_dtype` |
| `get_dispatch_dtype` | `(layer) -> DispatcherOutputDtype` | 读取上述标记，未设置时默认 `BF16` |
| `set_finalize_all_gather` | `(layer, enabled=True) -> None` | 标记 finalize 输出是否需要 TP all-gather（GGUF 专用） |
| `quant_mode_of` | `(dtype) -> int` | `bf16→-1`、`int8→1`、`mxfp8→3`；不支持的 dtype 报错 |
| `get_routing` | `(layer) -> (init, finalize, group_list_type)` | 按 `(dtype, all_gather)` 构建或命中 per-layer 缓存 |

常量：`_QUANT_MODE`（dtype→quant_mode 映射）、`GROUP_LIST_TYPE = 1`（`expert_tokens` 是计数而非前缀和的契约标记）。

#### （b）`layers/moe/moe_runner/ascend.py`

新增一对 permute 钩子，替换原 `("ascend_tp","ascend")` / `("ascend","ascend_tp")`：

```python
@register_pre_permute("standard", "ascend")
def pre_permute_standard_to_ascend(
    dispatch_output: StandardDispatchOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput: ...

@register_post_permute("ascend", "standard")
def post_permute_ascend_to_standard(
    runner_output: AscendRunnerOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput: ...
```

`running_state["ascend_routing"]` 的载荷（一次前向内有效，post 侧 `pop` 消费）：

```python
{
    "finalize": NPUFinalizeRouting | AllGatherFinalizeRoutingWrapper,
    "expanded_row_idx": Tensor,   # [T*K] int32，src→dst 置换
    "topk_ids": Tensor,           # [T, K] int32
    "topk_weights": Tensor,       # [T, K] bf16
}
```

#### （c）`hardware_backend/npu/quantization/moe_methods.py`

`_set_dispatcher_output_dtype` → **`_set_moe_dispatch_dtype`**：

```python
ascend_dispatch.set_dispatch_dtype(layer, dtype)                      # ① no-A2A 路径
dispatcher = getattr(layer, "dispatcher", None)
if dispatcher is not None and hasattr(dispatcher, "set_quant_config"):
    dispatcher.set_quant_config({"dispatcher_output_dtype": dtype})   # ② 传输层路径
```

7 处调用点（均在 `process_weights_after_loading` 内，且只对 `weight_prefix == "w13"`）：

| 方法 | dtype |
|---|---|
| `NPUW4A4MXFP4MoEMethod` | `bf16` |
| `NPUW4A4Int4MoEMethod` | `bf16` |
| `NPUW8A8Int8MoEMethod` | `int8` |
| `NPUW4A8Int8MoEMethod` | `int8` |
| `NPUWNA16Int4MoEMethod` | `bf16` |
| `NPUUnquantMoEMethod` | `bf16` |
| `NPUMXFP8MoEMethod` | `mxfp8`；DeepEP 时为 `bf16` |

#### （d）`layers/quantization/gguf.py`

```python
ascend_dispatch.set_finalize_all_gather(layer)      # 替代 set_quant_config({"quant_type": "gguf"})
```

#### （e）`layers/moe/fused_moe_triton/layer.py`

- 删除 `elif a2a_backend.is_none() and is_npu(): return AscendTPDispatcher(...)`；
- 新增 `_validate_ascend_no_a2a_layout(moe_runner_config)`，仅在 `_is_npu and a2a_backend.is_none()` 时调用。

---

## 3. 特性设计

### 3.1 抽象边界：dispatcher 管"通信/排布"，runner 的 IO 适配管"格式转换"

这是本次重构的判断依据。社区的 `MoeRunner` 用 `(dispatch_output.format, runner_backend)` 二元组查表选择 IO 适配器：

| 上游格式 | 算子族 | pre_permute 的工作量 |
|---|---|---|
| `standard` | triton / deep_gemm / aiter / humming | **在钩子里做排序**（社区一直如此） |
| `deepep_normal` | ascend | 只做装箱（重排已在 all-to-all 里完成） |
| `ascend_ops` | ascend | 只做装箱（重排由 MC2/AllToAll 完成） |
| **`standard`** | **ascend（本次新增）** | **做排序 + 量化 + 分组** |

也就是说：`("standard","ascend")` 不是新机制，而是**补上矩阵里原本空着的一格**。这与其它 backend 的做法完全一致，是本次设计"合理"的最强依据。

### 3.2 状态的两级拆分

旧实现把两类状态都放在 dispatcher 对象上（`self.init/self.finalize/self.quant_config/self._dispatch_output`），生命周长为"与 layer 同寿"，存在跨前向残留的风险。新设计按"是否跨前向"拆成两级：

| 容器 | 内容 | 生命周期 | 位置 |
|---|---|---|---|
| **layer 属性** | `moe_dispatch_dtype`、`moe_dispatch_finalize_all_gather`、`_ascend_routing_cache` | 与 layer 同寿（加载期写入） | `layer.__dict__` |
| **`running_state`** | `finalize`、`expanded_row_idx`、`topk_ids`、`topk_weights` | 一次 `MoeRunner.run()` | `run()` 的局部 dict |

收益：

- 配置只在加载期写一次，前向只读（热路径上是一次 `getattr` + 元组比较）；
- 每次前向的置换与权重随调用栈创建/销毁，异常路径不会留下脏状态；
- 与批重叠（TBO）天然兼容——每个 microbatch 有自己的 `running_state`。

`running_state` 是框架既有设施（`MoeRunner.run` 创建，同一次调用内传给 pre_permute / runner_core / post_permute），DeepEP 的 permute 也用它传 `topk_ids/topk_weights`，本设计沿用同一约定。

### 3.3 dtype 契约：把"权重需求"翻译成"routing 的输出格式"

一条 MoE 链路上有四个不同的 dtype 概念：

| # | 概念 | 例子 | 谁决定 |
|---|---|---|---|
| ① | 权重存储 dtype | `int8` / `fp8_e4m3` / `int4`（打包成 int32） | checkpoint + 量化配置 |
| ② | **dispatch dtype（激活）** | `bf16` / `int8` / `mxfp8` | **量化方法在加载期声明**（本次重构的契约） |
| ③ | scale dtype | 激活 scale 为 `fp32`（per-token）；权重 scale 为 `bf16`；MXFP8 为 e8m0 | 各算子签名 |
| ④ | 输出 dtype | `bf16` | `output_dtype=original_dtype` |

② 决定三件事：routing 产出的数据格式、是否带 scale、量化发生在哪里。映射关系：

| dispatch dtype | `quant_mode` | routing 产出 | scale | 谁做量化 |
|---|---|---|---|---|
| `bf16` | `-1` | bf16 | `None` | gmm1 内部（`hidden_states_quantizer`） |
| `int8` | `1` | int8 | `(T*K,)` fp32 | routing（重排时顺带） |
| `mxfp8` | `3` | e4m3 | e8m0 块 scale（reshape 为 pair-split） | routing（融合） |

关键点：

- ② **不等于**①。例如 W4A8（权重 int4）声明 `int8`，W4A4 声明 `bf16`，MXFP8 在 DeepEP 下声明 `bf16`；
- ② 只控制 **gmm1 之前** 的那次量化；gmm2 之前的量化由激活算子类型决定（W8A8 的 `NPUSwigluQuant` 恒产 int8）；
- 声明只在 `weight_prefix == "w13"` 时发生，因为 routing 的输出只喂 w13。

### 3.4 为什么 finalize 必须提前到 `post_permute`

社区 `StandardDispatcher.combine` 对 no-A2A 是**透传**（`(hidden_states,) = combine_input; return hidden_states`），它的契约是"收 `[T,H]`、还 `[T,H]`"。

旧 `AscendTPDispatcher.combine` 才负责 finalize。dispatcher 删除后，若仍把 `[T*K,H]` 交给 `combine`，没有任何环节会再收拢。因此 finalize 放在 `post_permute_ascend_to_standard`，直接产出 `StandardCombineInput([T,H])`。

### 3.5 GGUF 的 finalize all-gather：两个正交开关

GGUF 专家权重不按中间维切分，finalize 输出的 hidden 维在 TP 下不完整，需要补一次 all-gather。旧实现把这件事塞进 `dispatcher.quant_config` 字典（`{"quant_type": "gguf"}`），与 dtype 共用一个容器，存在两个问题：

- 字典是**覆盖式**写入（`self.quant_config = quant_config`），两个写入方会互相清掉；
- `set_ascend_dispatcher_output_dtype()` 会重建 finalize，若 dtype 在 gguf 之后写入，wrapper 会被冲掉（旧代码靠调用顺序恰好安全）。

新设计拆成独立开关：

```python
set_dispatch_dtype(layer, dtype)         # "产什么格式"
set_finalize_all_gather(layer)           # "要不要在 finalize 里再做一次 all-gather"
```

`get_routing()` 在一次构建中原子地组合 `(init, finalize)`，不存在顺序问题。条件 `all_gather and tp_size > 1` 与旧实现逐字一致（tp=1 时不包，避免无意义的集合通信）。

### 3.6 `none + ep_size > 1`：为什么只对 NPU 拒绝

| 平台 | 是否合法 | 原因 |
|---|---|---|
| GPU | ✅ | 各 rank 持有**同一批 token**（复制而非分发），只计算本地 expert（`StandardDispatcher` 会把全局 id 映射成本地 id、并把不属于本地的置 `-1` 丢弃），最后 all-reduce 各卡贡献。用算力换通信 |
| NPU（本 PR） | ❌ | Ascend routing 算子按**全局** expert id 组织分组（`expert_num = num_experts`），而本地权重只有 `num_local_experts` 份；且它不走 `StandardDispatcher` 的本地映射。结果必然是索引空间不匹配（kernel 报错或越界） |

因此 `_validate_ascend_no_a2a_layout` 只在 `NPU + none` 时生效：`num_local_experts != num_experts` 即报错（DWDP 例外，它在构造后把 EP 视图塌缩为全量权重）。这是把"运行期未定义行为"变成"建层期明确拒绝"，不改变任何原本可用的配置。

### 3.7 为什么路由逻辑放在 permute，而不是塞进 `AscendRunnerCore.run()`

1. **矩阵会失效**：`run()` 只应知道 `AscendRunnerInput`（hs / scale / expert_tokens / group_list_type），不应感知 `topk_output`；否则 deepep / ascend_ops 这两条"已经排好序"的路径无法复用同一个 core。
2. **类型会污染**：把 `topk_output` 塞进 `AscendRunnerInput`，等于让计算层感知排布层。
3. **职责单一**：`run()` 现在被 4 种上游共用（standard / deepep_normal / deepep_ll / ascend_ops），不应被 none 路径的特殊性绑住。

---

## 4. 方案设计

### 4.1 调用链对比

**变更前（NPU + none）**

```
FusedMoE.forward_impl
 ├─ AscendTPDispatcher.dispatch               # 做 routing：init_routing
 ├─ MoeRunner.run
 │    ├─ pre_permute("ascend_tp","ascend")     # 只装箱
 │    ├─ AscendRunnerCore.run                  # gmm1 → act → gmm2
 │    └─ post_permute("ascend","ascend_tp")    # 只装箱
 └─ AscendTPDispatcher.combine                # 做 finalize
```

**变更后（NPU + none，与 GPU 同一份 dispatcher）**

```
FusedMoE.forward_impl
 ├─ StandardDispatcher.dispatch               # 透传（ep=1 时不重映射）
 ├─ MoeRunner.run
 │    ├─ pre_permute("standard","ascend")      # ★ 做 routing：init_routing + 量化
 │    ├─ AscendRunnerCore.run                  # 不变
 │    └─ post_permute("ascend","standard")     # ★ 做 finalize
 └─ StandardDispatcher.combine                # 透传
```

### 4.2 关键流程：pre_permute 内部

```
MoeRunner.run                                  runner.py:195
 ├─ PermuteMethodPool.get_pre_permute("standard","ascend")   runner.py:241
 └─ pre_permute_standard_to_ascend             ascend.py:324
      ├─ TopKOutputChecker.format_is_standard(...)            校验 topk 格式
      ├─ 拒绝 hidden_states_pre_quant                         routing 自己量化，不能吃预量化输入
      ├─ _routing_layer(runner_config)                        ascend.py:359 → config.layer
      ├─ ascend_dispatch.get_routing(layer)                   ascend.py:365
      │    ├─ get_dispatch_dtype(layer) → "int8"              dispatch.py:103
      │    ├─ quant_mode_of("int8") → 1                       dispatch.py:109
      │    └─ NPUMoEInitRouting_v2(quant_mode=1)              命中缓存或新建
      ├─ init._init_routing(hs, ids, 64, 6)                   ascend.py:372
      │    └─ torch.ops.npu.npu_moe_init_routing_v2(...)      init_routing.py:102
      ├─ running_state["ascend_routing"] = {...}              ascend.py:379
      └─ return AscendRunnerInput(...)
```

### 4.3 状态与内核的生命周期

```
【加载期】量化方法 process_weights_after_loading
   └─ set_dispatch_dtype(layer, "int8")            → layer.moe_dispatch_dtype = "int8"
   └─ (GGUF) set_finalize_all_gather(layer)        → layer.moe_dispatch_finalize_all_gather = True

【首次前向】pre_permute
   └─ get_routing(layer) → 构建 (init, finalize, group_list_type) → 缓存到 layer._ascend_routing_cache

【稳态前向】pre_permute
   └─ get_routing(layer) → 命中缓存（一次 getattr + 元组比较）
```

### 4.4 文件改动清单（15 文件，+292 / −230）

| 文件 | 改动 | 原因 |
|---|---|---|
| `hardware_backend/npu/moe/dispatch.py` | **新增 116 行** | routing 的状态与内核工厂 |
| `hardware_backend/npu/moe/quantization/moe_methods.py` | helper 改名 + 双写（layer / transport dispatcher），7 处调用点 | dtype 契约的声明方 |
| `layers/moe/moe_runner/ascend.py` | 新增一对 permute，删除旧的一对 | 路由与收拢下沉到 IO 适配 |
| `layers/moe/fused_moe_triton/layer.py` | 删 NPU 专属分支；新增 EP 布局校验 | 总入口 |
| `layers/moe/token_dispatcher/ascend_tp.py` | **删除 143 行** | 职责已迁出 |
| `layers/moe/token_dispatcher/base.py` | 删枚举成员、两个 checker、TYPE_CHECKING 导入 | 格式名不再存在 |
| `layers/moe/token_dispatcher/__init__.py` | 删 3 个导出 | 同上 |
| `layers/moe/utils.py` | 删 `MoeA2ABackend.ASCEND_TP`、`is_ascend_tp()` | 同上 |
| `layers/quantization/gguf.py` | `set_finalize_all_gather(layer)` | 正交化 |
| `arg_groups/fields/exec_.py`、`fields/spec.py` | 删 `ascend_tp` 取值 | CLI 契约 |
| `arg_groups/moe_hook.py` | 删归一化块 | 不可达 |
| `docs/.../quantization.mdx`、`dits/hunyuan_image3.py` | 注释/文档改述 | 指向已删除的类 |
| `test/registered/unit/server_args/test_resolution_declarations.py` | shape 的 backend 值 `ascend_tp`→`none` | 该取值已不存在 |

### 4.5 实施顺序（依赖关系）

| 步骤 | 内容 | 为什么必须按此顺序 |
|---|---|---|
| W1 | 先新增 `("standard","ascend")` / `("ascend","standard")` 两个 permute | 若先切 dispatcher，`get_pre_permute` 返回 `None`，**第一次 MoE 前向**才 assert |
| W2 | 切 dispatcher 工厂 + 删除 `ascend_tp` 相关代码 | 依赖 W1 已就位 |
| W3 | dtype 从 dispatcher 字典挪到 layer（与 W2 同批） | 否则 `set_quant_config` 被静默存下但无人读取，routing 退回 BF16 → 数值/性能静默漂移 |
| W4 | （未做）把 runner 里 `is_deepep()` 的激活选择改为数据驱动 | 行为等价，属后续清理 |
| W5 | EP>1 显式拒绝 | 安全边界 |
| W6 | 测试/文档/注释 | 收尾 |

### 4.6 风险与对策

| 风险 | 后果 | 对策 / 现状 |
|---|---|---|
| dtype 未注入（静默降级为 bf16） | 数值漂移、显存翻倍、多一趟量化 | 7 处声明点已按旧实现逐一核对（7→7）；**未加运行期断言**（刻意，见 §5.7） |
| GGUF + TP>1 的 all-gather 丢失 | hidden 维不完整 | 标记搬到 layer；`get_routing` 内 `all_gather and tp>1` 与旧实现一致 |
| topk 格式不匹配 | `AttributeError` / 静默错 | `TopKOutputChecker.format_is_standard` 显式拒绝非标准格式 |
| CUDA/aclgraph 捕获 | 捕获期执行 Python、重放只跑 kernel | 新代码无 `.item()` / `.cpu()` / 数据相关分支，捕获安全；但**未在开图模式下验证**（见 §5.7） |
| 外部脚本传 `ascend_tp` | 启动报错 | 有意为之：从"静默改写"改为"显式失败" |
| 外部代码 import `AscendTPDispatcher` | `ImportError` | 仓库内已清零；下游项目需同步 |

---

## 5. 验证结果

### 5.1 环境与配置

| 项 | 值 |
|---|---|
| 硬件 | Ascend NPU（双卡机器） |
| 模型 | DeepSeek-V2-Lite-W8A8（ModelSlim 量化，64 routed experts，top-6，H=2048，moe_intermediate=1408） |
| 基线 | `feat/moe-trace` @ `1874e1d`（旧 `AscendTPDispatcher` + env 门控打点/dump） |
| 迁移 | `feat/moe-none-merge` @ `3e17228` |
| 配置 | tp=1（prefill/decode）、tp=2 + dp-attention（prefill） |
| 共性参数 | `--skip-server-warmup`、`--disable-cuda-graph`、`--disable-radix-cache` |

### 5.2 验证方法

在两层实现上跑**完全相同的请求**，对 MoE 段落盘的张量做逐比特比对：

- 打点/落盘工具：`debug_utils/moe_trace.py`（env 门控，只在基线分支存在），对第 1/13/26 层落 4 个文件：
  `dispatch_out.pt`（hs / scale / row_idx / expert_tokens / topk_ids / topk_weights）、`runner_out.pt`、`combine_out.pt`、`forward_impl_out.pt`；
- 请求：`/v1/chat/completions`，`temperature=0`；
  - `max_tokens=1` → 第一个 token 由 prefill 的 logits 采样，不产生 decode forward → 落盘为 **prefill**（9 token → 54 行）；
  - `max_tokens=2` → 产生 1 步 decode，落盘被覆盖为 **decode**（1 token → 6 行）；
- 判据：形状与 dtype 一致，且 `max|Δ| == 0`（浮点）/ `torch.equal == True`（整型），dump 文件 md5 相同。

### 5.3 数值比对结果

| 配置 | 路径 | 覆盖层 | 结果 |
|---|---|---|---|
| tp=1 | prefill（54 行） | L1 / L13 / L26 | **全部逐比特一致**，md5 相同 |
| tp=1 | decode（6 行） | L1 / L13 / L26 | **全部逐比特一致**，md5 相同 |
| tp=2 + dp-attention | prefill（全局 9 行，两个 rank 各自落盘） | L1 / L13 / L26 | **两 rank × 三层 × 四个张量全部逐比特一致** |

附加不变量（tp=2）：

- 两个 rank 的 `dispatch_out` **逐位相同**（同一全局 batch 上冗余计算 routing）→ 实证了 TP 正确性所依赖的"确定性"前提；
- 两个 rank 的 `runner_out` / `combine_out` 互不相同（各自只算一半中间维的部分和），跨分支同 rank 相同 → 说明比对是有意义的，不是空转。

### 5.4 单元测试

```bash
python3 -m pytest test/registered/unit/server_args/test_resolution_declarations.py -x -q
# 14 passed
```

该套件校验参数解析的"声明 → 投影"一致性。注意：其中一个用例原本用 `ascend_tp` 构造"两个 writer 冲突"的场景，取值改为 `none` 后该冲突不再产生，**该 shape 的覆盖语义有所减弱**（测试仍通过）。

### 5.5 服务冒烟

单卡起服务 + 一条 chat 请求（`max_tokens=8`）返回通顺中文，`finish_reason=length`，`prompt_tokens=9` 与打点中的 `T=9` 一致。

### 5.6 覆盖矩阵

| 维度 | 已覆盖 | 未覆盖 |
|---|---|---|
| 阶段 | prefill、decode | — |
| 并行 | tp=1、tp=2 + dp-attention | — |
| 量化 | W8A8-int8 | BF16、W4A8、W4A4、MXFP8、GGUF |
| 模型 | DeepSeek-V2-Lite | 其它 MoE 模型、多模态 DiT |
| 执行模式 | eager（`--disable-cuda-graph`） | aclgraph / CUDA graph 捕获后的重放路径 |
| 后端 | none | deepep / ascend_ops / ascend_fuseep（仅静态核对，未实测） |

### 5.7 已知未验证项与后续计划

| # | 项 | 建议验证方式 |
|---|---|---|
| 1 | BF16 档位 | 起 BF16 模型，确认 `dispatch_out.scale is None` |
| 2 | MXFP8 档位 | 确认 `quant_mode=3`、`scale` 为 pair-split 三维 |
| 3 | GGUF + TP>1 | 确认 finalize 内的 all-gather 生效、hidden 维正确 |
| 4 | 其它 MoE 模型 | 至少覆盖"融合 shared expert"与"non-gated"两类 |
| 5 | graph 捕获路径 | `--cuda-graph-bs-decode 1 2 4` 起服务，与 eager 输出逐字符比较 |
| 6 | DeepEP 回归（DeepSeek-V4-Flash） | 16 卡脚本冒烟；重点确认 `dispatcher_output_dtype` 仍以 `int8` 传给 DeepEP |
| 7 | dtype 未注入的静默降级 | 后续 PR 可在 `get_routing` 增加"权重已量化但 dtype 仍为 bf16"的断言 |
| 8 | runner 内 a2a 条件分支（W4） | 改为 `pertoken_scale is None` 的数据驱动判断，行为等价 |

---

## 附录：结论

- 本次重构把"token 按 expert 重排"从通信抽象（dispatcher）迁到了 IO 适配（runner 的 permute 钩子），使 NPU 的 `none` 分支与 GPU 走同一份 `StandardDispatcher`，并补上了 permute 矩阵中长期空缺的 `("standard","ascend")` 一格；
- 迁移是**搬运而非改写**：routing / grouped GEMM / activation / finalize 的调用参数逐个未变，三个配置的逐比特比对为此提供了直接证据；
- 行为变化仅两处，均为"把静默行为显式化"：删除 `ascend_tp` CLI 取值（启动报错）、拒绝 `none + ep_size>1`（建层报错）。
