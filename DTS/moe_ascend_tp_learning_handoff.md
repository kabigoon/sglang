# 交接文档 v2：Ascend MoE 的 `ascend_tp → none` 合并（学习 · 重构 · 验收）

> **给新窗口的 agent**：请先完整读完本文，再按 §1 的角色设定和 §3 的进度推进。
> 本文是自包含的：你不需要（也无法）看到前几个窗口的对话。
> 生成时间：2026-10-23（v1 为 2026-10-21）。**本文档已经过一次实测校正**：v1 里有 3 条结论是错的（见 §9 标注 ❌→✅），别再用旧说法。
> 行号会随分支漂移：**一律用 `grep -n` 定位，不要背行号**。

---

## 0. 一句话背景与当前状态

**任务**：把 Ascend 专属的 `ascend_tp` MoE dispatch 逻辑合并进社区 `none` 路径（token 按 expert 重排 **不是通信语义**，不该占用 dispatcher 抽象）。

**已经完成的三个阶段**：

1. **学懂**（v1 窗口）：用 env 门控的 `moe_trace` 打点 + dump，把 MoE 全链路逐行走了一遍，推翻了 v1 文档里 3 条错误结论。
2. **重构 + 验收**（v2 窗口）：
   - 在新 base（`chenxu214/ci_sglang` 的 `feat/dev_ep`，相对共同祖先 `69076b9` 有 22 个提交）上重新落了迁移，产出分支 **`feat/moe-none-merge`**，已开 **PR #8**。
   - 数值验收：tp=1 的 prefill/decode + tp=2 的 prefill，**三个配置全部逐比特一致**（dump 文件 md5 相同）。
3. **待办**：其它量化档位 / 其它模型的冒烟、DSV4-Flash(deepep) 冒烟结果确认、PR 评审。

**当前工作重点**：**守 PR + 补覆盖**，不是再改架构。

### 0.1 原始需求与验收标准（来自旧任务书，仍然有效）

> **验收标准（原文）**：在不配置 `moe-a2a-backend` 参数时，链路保持原有 `ascend-tp` 逻辑不变。

总体设计给出的四条方向（决定"什么该动、什么不该动"）：

1. `ascend_tp` backend 与社区 `None` backend **合并**；Ascend 的差异保留在 **MoE Runner / 算子实现**中；
2. CANN 专用的 dispatch 能力逐步通过 **`ascend-ops`** 接入（覆盖 prefill、decode、不同机型组网）——**本 PR 不碰**；
3. DeepEP backend 复用社区接口（便于后续接 DeepEP-Ascend、ZBAL）；
4. 动态 EPLB / 不同 EP 库切换是后续工作；本次先保证 `none` 默认链路行为兼容。

**这不是"把字符串 `ascend_tp` 改名为 `none`"**：参数层早就归一了，任务要的是**结构收敛 + 用测试锁住行为**。

---

## 1. 你的角色与工作方式（重要）

你是一位**教学式结伴分析者 + 迁移验收搭档**。目标有两个：

- 让用户能自己讲清这条链路（他现在的水平：能独立解释 dtype 契约、running_state、permute 矩阵，见 §3）；
- 让每一个结论都有**可复现的证据**（日志行 / dump 比对 / 单测），而不是"看起来对"。

必须遵守：

1. **结论必须带证据**：每个论断给 `file:line` 或用户回传的日志/dump；不确定就写"待实测"。
2. **一次只给一个动作**：给一个可执行命令 + 一个明确的问题，等回传。
3. **先解释、再推进**：拿到日志先把数字和代码对上（形状、dtype、含义），再给下一步。
4. **用户不能让你跑代码**：本地 Mac 没有 torch / 没有 NPU。所有运行由用户在 NPU 服务器上做（远程机器可能多人共用，跑前先 `ss -ltnp | grep <port>`）。
5. **改代码只在必要时**：主干分支不允许留调试代码（打点、断言）；调试用临时分支或 `feat/moe-trace`。
6. **不要重复推导 §9**：那些已有实测证据，直接用。
7. **诚实优先于自信**：v1 窗口在 3 个地方"讲得很确定但其实是错的"（§9 全部标注）。发现旧结论与自己读到的不符时，**以实测为准，并在回复里明确更正**。

---

## 2. 环境与分支

| 项 | 值 |
|---|---|
| 本地工作区 | `/Users/an/Developer/sglang`（**无 torch，不能运行**） |
| 远端 NPU 机器 | 用户 Linux，多卡（已跑过 tp=1 / tp=2，DSV4 脚本用 16 卡） |
| 模型 | `/home/a00821909/DeepSeek-V2-Lite-W8A8`（W8A8，ModelSlim 量化） |
| DSV4 模型 | `/home/weights/DeepSeek-V4-Flash-0731-w8a8`（16 卡脚本，见 §8.6） |
| 远端仓库 | `origin` = `git@github.com:kabigoon/sglang.git`（用户的 fork） |

### 分支地图（**这是最容易搞混的地方**）

| 分支 | 内容 | 用途 |
|---|---|---|
| `feat/dev_ep` @ `chenxu214/ci_sglang` = `d3c76503` | 同事的分支（相对共同祖先 `69076b9` 共 **22 个提交**，含他新加的 **`ascend_ops` dispatcher**） | **迁移的正确 base** |
| **`feat/moe-none-merge`** @ `origin` = `3e17228f` | **本次 PR #8 的分支**：迁移 + 去调试 + 删 `ascend_tp` | 主线 |
| `feat/moe-trace` @ `origin` = `1874e1d8` | 旧代码 + `moe_trace` 打点 + `mlp_*` 边界打点 | **调试/学习基线**（有打点，别当主线） |
| `feat/ascend-tp-merge-none` @ `origin` = `7d3ba673` | 旧 base 上的迁移初版（含打点、含 `ascend_tp` 兼容别名） | 历史参考，**不要往主干合** |

三者关系：

```
69076b9 (上游)
 ├─ b8ea873  ← feat/moe-trace 的祖先（打点提交）
 │    └─ 1874e1d  (+ mlp_* 边界打点、fmt 修复)          ← 调试基线
 │
 └─ d3c7650  ← chenxu214/feat/dev_ep（同事 22 个提交）
      └─ 3e17228  ← feat/moe-none-merge（本 PR，单提交，15 文件）   ← 主线
```

> **PR 是跨 fork 的**：base 是 `chenxu214/ci_sglang:feat/dev_ep`，head 是 `kabigoon/sglang:feat/moe-none-merge`。
> 在**自己的**仓库点 "Compare & pull request" **永远看不到 chenxu214 作为 base**（它只列你有权限的仓库）。
> 正确入口：直接构造 compare URL，或从 **目标仓库** 的 compare 页走 "compare across forks"：
> `https://github.com/chenxu214/ci_sglang/compare/feat/dev_ep...kabigoon:feat/moe-none-merge`

---

## 3. 进度与下一步（替代 v1 的"学习目标"）

### 已完成（用户已经掌握，不用再教）

| 主题 | 用户的掌握程度 |
|---|---|
| 一次 MoE 的完整链路（gate → dispatch → gmm1 → act → gmm2 → combine → all-reduce/reduce-scatter） | ✅ 能自己讲 |
| routing 的 4 个返回值（hs / scale / row_idx / expert_tokens） | ✅ 能讲，且知道我们用 dump 实测过 |
| `quant_mode` 三种取值、谁决定、在哪生效 | ✅ 能讲 |
| dtype 的"四个概念"（权重 / dispatch / scale / 输出）与"契约"本质 | ✅ 能讲 |
| `running_state`（创建 → 三站传递 → pop 消费） | ✅ 能讲 |
| tp>1 的通信（DP gather / TP reduce-scatter / 为什么不是 A2A） | ✅ 能讲 |
| 重构后的结构（dispatch.py 工厂 + 两个 permute 钩子 + 8 处 dtype 注入） | ✅ 能逐文件评审 |

### 待办（按优先级）

1. **确认 DSV4-Flash + deepep 冒烟结果**（用户在 v2 末尾准备跑，结果未回传）。这是"其它后端未受影响"的最强证据。
2. **三个量化档位冒烟**（各 30 秒）：BF16（验 `scale=None`）、MXFP8（验 `quant_mode=3` + pair-split）、GGUF+TP2（验 finalize 里的 all-gather）。
3. **PR #8 评审跟进**：处理 reviewer 意见；尤其是"只验证了 W8A8"这点要如实写清。
4. （可选）给 `get_routing()` 加一条"dtype 未注入"的兜底断言——**用户已明确说过主干分支不要防御性断言**，若要做，放到后续 PR。

---

## 4. 背景知识（30 秒版，已按实测修正）

MoE 每层对每个 token：`y_t = Σ_{k=1..K} w[t,k] · Expert_{e[t,k]}(x_t)`（V2-Lite：`K=6`，64 routed experts，**shared expert 不融合**）。

实现不做"每 token 循环"，而是**把 (token, expert) 对摊平成 `T*K` 行**，按 expert 升序重排后一次 grouped GEMM 算完，再散回加权：

```
dispatch : npu_moe_init_routing_v2(x, topk_ids, expert_num=E, quant_mode=q)
             -> hs [T*K, H]      按 expert 升序排好的行（可能已量化）
             -> expanded_row_idx [T*K]   ★ src→dst 置换（不是"行属于哪个 token"）
             -> expert_tokens [E]        ★ 每个 expert 的**计数**（不是累加和）
             -> pertoken_scale           int8: [T*K] fp32；mxfp8: e8m0 块 scale；bf16: None

combine  : npu_moe_finalize_routing(hs, scales=topk_weights,
                                    expanded_src_to_dst_row=row_idx,
                                    export_for_source_row=topk_ids, drop_pad_mode=2)
             -> [T, H]             加权求和 + 搬回原位置
```

`quant_mode`（`hardware_backend/npu/moe/init_routing.py`）由**权重量化方法**决定，不是 flag：

| 权重档位 | dispatch dtype | quant_mode | scale |
|---|---|---|---|
| BF16 | `bf16` | `-1` | 恒为 `None` |
| W8A8 / W4A8-int8 | `int8` | `1` | 每行一个 **fp32** |
| MXFP8 / W4A8-MXFP4 | `mxfp8` | `3`（`MXFP8_QUANT_MODE`） | e8m0 块 scale，reshape 成 pair-split `[N, M/2, 2]` |

---

## 5. 重构后的架构地图（**新窗口必读**）

### 5.0 三层概念：别把"通信后端"和"计算 kernel"混为一谈

| 层 | 参数 | 实现 | 本 PR 是否动 |
|---|---|---|---|
| **参数/解析层** | `--moe-a2a-backend` | `arg_groups/fields/exec_.py` + `arg_groups/moe_hook.py` | 删 `ascend_tp` 取值与归一化 |
| **调度层（通信/排布）** | 同上 → `MoeA2ABackend` | `create_moe_dispatcher()` + `token_dispatcher/*` | **主战场** |
| **计算层（expert GEMM）** | `--moe-runner-backend` | `MoeRunnerBackend.ASCEND` → `AscendRunnerCore` | **不动** |

两条容易搞错的事实：

1. **NPU 上 `MoeRunnerBackend.ASCEND` 由量化方法自己决定**，与 a2a backend 无关（各 quant method 的 `create_moe_runner` 里 `backend = MoeRunnerBackend.ASCEND`）。所以"差异保留在 Runner/算子层"今天基本已经成立。
2. **`none` ≠ 没有 routing**：社区 `none` 不做显式 token A2A，但仍然要按 top-k 选出 expert；ep=1 时每个 rank 都有全部 expert、只是权重被 TP 切窄。

**布局公式**（`runtime_context.py:154`）：

```
moe_tp_size = tp_size // moe_ep_size // moe_dp_size
```

| 配置 | 结果 | 含义 |
|---|---|---|
| `tp=2, ep=1` | `moe_tp=2` | 每卡有**全部** 64 专家，每个专家的 intermediate 切一半（1408→704） |
| `tp=2, ep=2` | `moe_tp=1` | 每卡放**一部分完整**专家（需要 A2A 才能把 token 送过去） |

**DeepEP 会强制 `ep_size = tp_size`**（`overrides.py::_a2a_ep_size`），所以 `tp=2` 时写 `--ep-size 1` 也会被调成 2——不要把它当成独立的运行模式。

### 5.1 调用链的三个角色（dispatcher → MoeRunner → RunnerCore）

```
dispatcher（通信抽象）
  none → StandardDispatcher（纯透传）        ← 本 PR 后 NPU 也走这里
  ascend_ops → AscendOpsDispatcher（同事新加，真 A2A，未动）
  deepep/pplx/mori/nixl/mooncake → 各自实现（未动）

MoeRunner（(格式, 算子族) 矩阵 + IO 适配）
  ("standard","ascend") → pre_permute_standard_to_ascend    ← 本 PR 新增
  ("ascend","standard") → post_permute_ascend_to_standard    ← 本 PR 新增
  ("ascend_ops"/"deepep_normal"/"deepep_ll","ascend")      ← 未动

AscendRunnerCore（算）：gmm1 → activation → gmm2            ← 一行未改
```

### 5.2 关键文件（新分支）

| 文件 | 职责 |
|---|---|
| `hardware_backend/npu/moe/dispatch.py`（**新增 116 行**） | routing 的**状态 + 内核工厂**：`set_dispatch_dtype` / `set_finalize_all_gather` / `get_routing`（按 `(dtype, all_gather)` 缓存） |
| `layers/moe/moe_runner/ascend.py` | 两个钩子：pre 里跑 `init_routing`（排序+量化+分组），post 里跑 `finalize`（散回+加权）；状态走 `running_state["ascend_routing"]` |
| `hardware_backend/npu/quantization/moe_methods.py` | **dtype 的声明方**：`_set_moe_dispatch_dtype`（写 layer **并**转发给 transport dispatcher），7 个量化档位各一处 |
| `layers/moe/fused_moe_triton/layer.py` | dispatcher 选择：删掉 `none+NPU → AscendTPDispatcher`；新增 `_validate_ascend_no_a2a_layout`（ep>1 报错） |
| `layers/quantization/gguf.py` | `set_finalize_all_gather(layer)`（GGUF 全量权重 → finalize 需 all-gather） |
| `token_dispatcher/ascend_tp.py` | **已删除**（143 行，遗物分配见 §5.4） |

### 5.3 两个"状态容器"的分工（这是重构的核心设计）

| 容器 | 存什么 | 生命周期 | 位置 |
|---|---|---|---|
| `layer.moe_dispatch_dtype` / `moe_dispatch_finalize_all_gather` / `_ascend_routing_cache` | 跨前向**不变**的配置 | 与 layer 同寿 | layer 属性 |
| `running_state["ascend_routing"]` | 一次前向内的**置换与权重**（`expanded_row_idx` / `topk_ids` / `topk_weights` / `finalize`） | 一次 `MoeRunner.run()` | `run()` 里的局部 dict |

`running_state` 由 `MoeRunner.run()` 创建（`runner.py` 里 `running_state = {}`），三站传递：`pre_permute(...)` → `runner_core.run(...)` → `post_permute(...)`；消费方用 **`pop`**（一次性）。框架里早有先例：`down_gemm_overlap_args`（GPU overlap）、deepep 的 `topk_ids/topk_weights`。

### 5.4 被删文件的"遗物分配表"

| 旧 `ascend_tp.py` 的部分 | 新归宿 |
|---|---|
| `AscendTPDispatchOutput`（7 字段） | `StandardDispatchOutput`（hs/scale/topk_output）+ `running_state`（row_idx/ids/weights）+ `AscendRunnerInput`（hs/scale/expert_tokens/group_list_type） |
| `AscendTPCombineInput` | `StandardCombineInput`（**已收拢的 [T,H]**） |
| `__init__` 的 `num_experts`/`top_k` 快照 | 每次前向读 `runner_config` |
| `set_ascend_dispatcher_output_dtype()`（3 个 if 分支） | `dispatch.py` 的 `_QUANT_MODE` 表 + `quant_mode_of()` + `get_routing()` |
| `set_quant_config()`（dtype + GGUF 包装） | 拆成 `set_dispatch_dtype()` + `set_finalize_all_gather()` 两个独立入口 |
| `dispatch()` | `pre_permute_standard_to_ascend()` |
| `combine()` | `post_permute_ascend_to_standard()`（**finalize 提前到这里**，因为社区 combine 是透传） |
| `self._dispatch_output` | `running_state["ascend_routing"]` |

---

## 6. 打点工具与"打点必须打在真正执行的路径上"

文件：`python/sglang/srt/debug_utils/moe_trace.py`（**只在 `feat/moe-trace` 上有**，主干分支已全部移除）。

机制：env 门控；不设 `SGLANG_MOE_TRACE` 时 `trace()/dump()` 立即返回。每个 TP rank 写自己的 `rank_tp{N}.log`；`dump()` 落 `.pt` 到 `<DIR>/dump/tp{N}/L{layer}/{tag}.pt`。

```bash
export SGLANG_MOE_TRACE=1
export SGLANG_MOE_TRACE_LAYERS=1          # 只打某几层（不设=所有层）
export SGLANG_MOE_TRACE_LIMIT=2           # 每个 (tag, layer, rank) 记几次；0=不限
export SGLANG_MOE_DUMP_LAYERS=1,13,26     # 只对这些层落张量
export SGLANG_MOE_TRACE_DIR=/tmp/moe_trace
```

### 6.1 tag 表（**已按新分支修正**）

| tag | 位置 | 说明 |
|---|---|---|
| `dispatcher` / `dispatcher_chosen` | `create_moe_dispatcher` | `a2a`、`E/E_local/top_k/H`、`moe_tp/moe_ep`、`quant=`、`cls=` |
| `set_dispatcher_dtype`（旧）/ `set_moe_dispatch_dtype`（迁移版） | 量化方法 | dtype 声明点 |
| `ascend_tp_init` / `ascend_tp_quant_config` | 旧 dispatcher | **迁移后消失** |
| `forward_impl_in` / `forward_impl_out` | 每层 MoE 入口/出口 | `forward_impl_out` 是 **all-reduce 之后**（若发生） |
| `dispatch_in` / `dispatch_out` | 旧：dispatcher；**新：pre_permute 里** | 见 5.2 |
| `moe_runner` / `pre_permute` / `post_permute` | `MoeRunner.run` | **不带 `layer=`**，所以只受 `LIMIT` 约束，会被截断 |
| `runner_in` / `runner_mid` / `runner_out` | `AscendRunnerCore.run` | gmm1 前 / act 后 / gmm2 后 |
| `combine_io` | combine（新分支：post_permute 里） | `[T*K,H] → [T,H]` |
| `mlp_in_local` / `mlp_in_global` / `mlp_out_global` / `mlp_out_local` | `DeepseekV2DecoderLayer.forward`（v2 新增） | **DP gather 前 / 后、切行前 / 后**；能看到 `(0,2048)` 的空转卡 |
| ~~`transport`~~ | ❌ **V2-Lite 不会出现** | `MoeRowTransport` 只在 **V4** 构造（`deepseek_v4_platform_stage.py`）；V2 走 `layer_boundary`（`ffn_boundary.prepare/complete_now`） |

### 6.2 两个"打点打错路径"的教训（**极其重要**）

| 打在哪 | 为什么没出现 | 判据 |
|---|---|---|
| `moe_transport.py` 的 `transport` | 只有 **DeepSeek-V4** 构造 `MoeRowTransport` | V2 用 `layer_boundary` |
| `op_comm_prepare_mlp` / `op_comm_postprocess_layer` | 只在**批重叠（TBO/SBO）**路径被调用（`batch_overlap/operations_strategy.py`） | 默认路径在 `DeepseekV2DecoderLayer.forward` 里内联做 gather/cut |

**规则**：打点前先 `grep -rn "函数名"` 确认调用点；跑完没日志 → 先怀疑路径，不要怀疑环境。

---

## 7. 路线（六阶段，前三阶段已完成）

### 7.0 实施顺序（已验证过的顺序，照此可避免"第一个 forward 就 assert"）

| # | 步骤 | 状态 |
|---|---|---|
| W1 | **先**新增 `("standard","ascend")` / `("ascend","standard")` 两个 permute（纯新增，不动旧路径） | ✅ |
| W2 | 再切 dispatcher 工厂（删 `none+NPU` 分支）+ 删除 `ascend_tp.py` / 枚举 / 导出 | ✅ |
| W3 | **与 W2 同批**把 dtype 从 `dispatcher.quant_config` 挪到 layer（否则 int8/mxfp8 静默降级） | ✅ |
| W4 | 清掉 runner 里残留的 a2a 条件分支（把 activation 选择改成**数据驱动**） | ⬜ **未做**（保持 `is_deepep() or is_ascend_ops()`，行为不变） |
| W5 | 决定 EP>1 的行为 | ✅ 选"显式报错" |
| W6 | 测试 / 文档 / 注释 | ✅（除 §10 列出的覆盖缺口） |

> **W1 必须在 W2 之前**：permute 池里没有对应条目时，`MoeRunner.run` 会在**第一次 MoE 前向**才 assert（`PermuteMethodPool.get_pre_permute` 返回 None）。先加钩子、再切 dispatcher，顺序反了会得到一个"能起服务、一请求就崩"的分支。

> **W3 不能落后于 W2**：`BaseDispatcher.set_quant_config` 存在但没人读，配置会被**静默存下**然后被忽略 → init routing 退回 BF16。这是本次重构最隐蔽的回归点。

### Stage 0 ✅ 钉死参数落点
- 单卡起服务，看 `dispatcher` / `dispatcher_chosen`。
- **v1 通过标志写错**：实际是 `quant='ModelSlimFusedMoEMethod'`（外层壳），不是 `NPUW8A8Int8MoEMethod`；int8 的证据在 `set_dispatcher_dtype ... w13='NPUW8A8Int8MoEMethod'`。
- 结论：`a2a=none npu=True E=64 E_local=64 top_k=6 H=2048 moe_tp=1` + `cls=AscendTPDispatcher`（旧分支）/ `StandardDispatcher`（新分支）。
- `E_local` 按 **EP** 切（`num_local_experts = num_global_routed // moe_ep_size + fused_shared`），**不按 TP**。

### Stage 1 ✅ routing 的 4 个返回值（已实测，v1 有两处错）
- 详见 §9 的 #2 / #3。
- `dispatch_out` 的 `scale=(T*K,)` **fp32**；同一 token 的 6 行 `hs` 与 `scale` **逐位相同**。

### Stage 2 ✅ int8 数据流
- `routing 量化 → gmm1 → act(再量化) → gmm2`；`act_out=(T*K, I/2)`（tp=2 时 1408/2=704）。
- 一次 MoE 有**两次 A 侧量化**：第一次由 `moe_dispatch_dtype` 决定，第二次由**激活算子类型**决定（W8A8 的 `NPUSwigluQuant` 恒产 int8）。

### Stage 3 ✅ permute 池（迁移落点）
- `("standard","ascend")` / `("ascend","standard")` 这对钩子就是迁移的落点。
- 旧 `pre_permute` 只做"装箱"（零计算）；迁移后它开始**真干活**（routing）。
- 社区 `combine` 对 none 是**透传**，所以 finalize 必须搬到 `post_permute`。

### Stage 4 ✅ TP + dp-attention 的通信
- `mlp_in_local` / `mlp_in_global` 证明 gather；`OwedOutput` 证明 reduce-scatter **延迟执行**。
- **`tp>1` 的通信不是 A2A**：DP gather（复制）+ TP reduce-scatter（求和+切行）。A2A 是"由路由驱动的变长 token 交换"，`ascend_tp` 里没有。
- `expert_tokens` 两卡都是 64；两卡 `dispatch_out` 逐位相同 → 证明无 EP 分发。

### Stage 5 ⬜ 横向对照（deepep）
- 尚未做。价值：证明"deepep 与 none 共享 runner / 量化 / activation，差异只在上游"。
- 更省钱的替代：跑 **DSV4-Flash + deepep** 冒烟（§8.6），验证"其它后端未受影响"。

### Stage 6 ✅ 迁移前后逐张量比对（已通过）
- 三个配置全部逐比特一致，见 §12。

---

## 8. 命令手册

### 8.1 起服务（tp=1，最小可读）

```bash
rm -rf /tmp/moe_trace
export SGLANG_MOE_TRACE=1 SGLANG_MOE_TRACE_LAYERS=1 SGLANG_MOE_TRACE_LIMIT=2
export SGLANG_MOE_DUMP_LAYERS=1,13,26 SGLANG_MOE_TRACE_DIR=/tmp/moe_trace

sglang serve --model-path /home/a00821909/DeepSeek-V2-Lite-W8A8 \
  --trust-remote-code --tp 1 --device npu --attention-backend ascend \
  --skip-server-warmup --disable-cuda-graph --disable-radix-cache --log-level info \
  --max-total-tokens 4500 --chunked-prefill-size 256 --host 127.0.0.1 --port 5684
```

### 8.1b 零改码观测：模块级 dump（不需要打点）

SGLang 自带一个模块级 dump，可以在**完全不改代码**的情况下做 A/B：

```bash
--debug-tensor-dump-output-folder /tmp/tdup --debug-tensor-dump-layers 1,13
```

它会把**每个叶子模块**的输出存成 `/tmp/tdup/TP{rank}_.../Pass{N}.pt`（一个 `{模块名: tensor}` 的 dict），包括 `model.layers.{L}.mlp.gate` / `.topk` / `.experts`。

- ✅ 适合：迁移前后对同一 batch 做**模块级** A/B，尤其看 `mlp.experts` 的输出；
- ❌ 看不到：dispatch 内部的 `row_idx` / `expert_tokens` / `scale`——那些只能靠 `moe_trace` 的 `dump()`。

两段 dump 的对比脚本很简单：`torch.load` 两边，按 key 求 `max|Δ|`。

### 8.2 触发请求（**必须用 chat completions + 显式采样参数**）

```bash
curl -s http://127.0.0.1:5684/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V2-Lite-W8A8","messages":[{"role":"user","content":"你好"}],
       "temperature":0,"max_tokens":1,"stream":false}'
```

- **`max_tokens=1` → 只有 prefill**（第一个 token 由 prefill 的 logits 采样，不跑 decode）→ dump 留下 **`T*K` 行**那份。
- `max_tokens=2` → 跑 1 步 decode → dump 被覆盖成 **`T=1`、6 行**那份。
- **不要用 `/generate` 的顶层 `max_new_tokens`**：`GenerateReqInput` 只认 `sampling_params` 里的字段，顶层会被**静默忽略** → 回落到模型 `generation_config`（temperature 0.3）→ 答案每次不同、生成到 EOS。

### 8.3 读 dump

```bash
python3 - <<'PY'
import torch
d = torch.load("/tmp/moe_trace/dump/tp0/L1/dispatch_out.pt", map_location="cpu")
hs, sc, ridx, et, ids = d["hs"], d["scale"], d["row_idx"], d["expert_tokens"], d["topk_ids"]
print("hs", tuple(hs.shape), hs.dtype, "| scale", tuple(sc.shape) if sc is not None else None)
print("expert_tokens sum =", et.sum().item(), "last =", et[-1].item(), "monotone =", bool((et[1:] >= et[:-1]).all()))
print("row_idx =", ridx.tolist()[:20])
flat = ids.flatten().long()
bc = torch.bincount(flat, minlength=et.numel())
order = torch.argsort(flat, stable=True); rank = torch.empty_like(order); rank[order] = torch.arange(len(order))
print("expert_tokens == bincount(topk_ids):", bool((bc == et).all()))
print("row_idx == rank(topk_ids) (src->dst):", bool((rank == ridx.long()).all()))
PY
```

### 8.4 A/B 验收（dump 比对，**最硬的证据**）

```bash
# A: 基线（feat/moe-trace，旧 dispatcher）
cd /workspace/sglang && git checkout feat/moe-trace && git pull
rm -rf /tmp/golden && export SGLANG_MOE_TRACE_DIR=/tmp/golden   # 其余 env 同上
# 起服务 + 打同一个请求（§8.2），然后：
md5sum /tmp/golden/dump/tp0/*/*.pt | tee /tmp/golden_md5.txt

# B: 迁移版（feat/moe-none-merge，StandardDispatcher + permute 钩子）
git checkout feat/moe-none-merge && git pull
rm -rf /tmp/new && export SGLANG_MOE_TRACE_DIR=/tmp/new
# 同样的 serve 命令 + 同样的 curl

python3 - <<'PY'
import os, torch
A, B = "/tmp/golden/dump/tp0", "/tmp/new/dump/tp0"
bad = 0
for layer in sorted(os.listdir(A)):
    for name in sorted(os.listdir(os.path.join(A, layer))):
        a = torch.load(os.path.join(A, layer, name), map_location="cpu")
        b = torch.load(os.path.join(B, layer, name), map_location="cpu")
        for k in sorted(set(a) | set(b)):
            x, y = a[k], b[k]
            if not isinstance(x, torch.Tensor): continue
            if tuple(x.shape) != tuple(y.shape):
                print(f"{layer}/{name}.{k}: SHAPE {tuple(x.shape)} vs {tuple(y.shape)}"); bad += 1; continue
            if x.dtype.is_floating_point:
                d = (x.float() - y.float()).abs().max().item(); ok = d == 0
            else:
                ok = bool(torch.equal(x, y))
            print(f"{layer}/{name}.{k}: {'✓' if ok else '✗'} {tuple(x.shape)}")
            bad += 0 if ok else 1
print("\n全部逐位相同 ✅" if bad == 0 else f"\n存在 {bad} 处差异 ❌")
PY
```

**tp=2 的额外规则**：只能"**同 rank 跨分支**"比对。`dispatch_out` 两卡应相同（同一全局 batch），`runner_out`/`combine_out` 两卡**必然不同**（各算一半中间维的部分和）。

### 8.5 单元测试（CPU，秒级）

```bash
python3 -m pytest test/registered/unit/server_args/test_resolution_declarations.py -x -q
# 期望 14 passed
```

### 8.6 DSV4-Flash（16 卡，deepep）冒烟

用户脚本走 `--moe-a2a-backend deepep --deepep-mode auto` + DSPARK 投机 + `--tp-size 16 --dp-size 16 --enable-dp-attention`。要点：

- 这条路径**完全不经过**本 PR 改的 none 分支；
- 唯一相关的改动是 `moe_methods.py` 里"dtype 同时转发给 transport dispatcher"（否则 NPU DeepEP 会从 int8 静默退回 bf16）；
- 看三件事：能起、能出词、日志里没有 `Expert parallelism ... not supported`（那条只在 `none + ep>1` 触发）。

### 8.7 形状速查表（tp=1 / tp=2 对照，用来一眼判断日志是否合理）

| 量 | tp=1（T=7） | tp=2（T_global=9，dp-attention） |
|---|---|---|
| 进 MoE 的 `x` | `(7, 2048)` bf16 | `(9, 2048)` bf16（两卡相同，`mlp_in_local` 分别 9 / 0） |
| `topk_weights` / `ids` | `(7, 6)` bf16 / int32 | `(9, 6)` bf16 / int32 |
| permute 后 `hs` | `(42, 2048)` **int8** | `(54, 2048)` int8 |
| `scale` | `(42,)` **fp32** | `(54,)` fp32 |
| `expert_tokens` | `(64,)` int64，`sum = 42` | `(64,)`，`sum = 54` |
| gmm1 输出 | `(42, 2816)` bf16（2×1408，**推断值**） | `(54, 1408)` bf16（2×704，**推断值**） |
| act 输出 | `(42, 1408)` int8 + `pertoken_scale (42,)` | `(54, **704**)` int8 + `(54,)` |
| gmm2 输出 | `(42, 2048)` bf16 | `(54, 2048)` bf16 |
| finalize 输出 | `(7, 2048)` bf16 | `(9, 2048)` bf16（**切行前**，之后 reduce-scatter 到本卡行） |

> 记两条：**`T*K` 是行数不是 token 数**；**tp=2 时 intermediate 是 704，不是 1408**。

---

## 9. 已确认事实（**v1 里标 ❌ 的是错的，别再重复**）

1. `--moe-a2a-backend none` 和 `ascend_tp` 在 NPU 上最终都走同一个 dispatcher（旧：`AscendTPDispatcher`；新：`StandardDispatcher`）：`moe_hook.py` 把 `ascend_tp` 改写成 `none`，`layer.py` 再按 `none + is_npu` 选择。**新分支已经把 `ascend_tp` 从 CLI/枚举彻底删除**。
2. ✅ **（❌ v1 说"累加和"）`expert_tokens` 是每 expert 的"计数"**，等价于 `bincount(topk_ids)`。证据：dump 里 `sum == T*K`、最后一项为 0、不单调。
3. ✅ **（❌ v1 说"每行原属哪个 token"）`expanded_row_idx` 是 src→dst 置换**：`row_idx[src] = dst`，`src = t*K + k`。证据：`row_idx == rank(topk_ids)`（T=1 时是 `0..K-1` 的置换，不是全 0）。`finalize` 的参数名就是 `expanded_src_to_dst_row`。
4. W8A8 → `NPUW8A8Int8MoEMethod`（作为 w13/w2 的 kernel）→ 声明 `dtype="int8"` → `quant_mode=1` → `scale` 非空（`(T*K,)` fp32）。注意外层 method 是 `ModelSlimFusedMoEMethod`。
5. dp-attention 下 MoE 输入是 **DP gather 后的全局 token 批**；触发条件是 `attn_dp_size>1 and get_moe_a2a_backend().is_none()`。**实现位置不是 `moe_transport.py`**（那是 V4），而是 `layers/layer_boundary`（`ffn_boundary.prepare` → `dp_gather_replicate`）。
6. `--dp 2 --enable-dp-attention` 解析为 `attn_dp_size=2, dp_size=1` → `attn_tp_size=1`、`moe_tp_size=tp`。
7. `FusedMoE.reduce_results` **默认 False**，所以 `forward_impl` 里那次 `all_reduce` **不执行**；TP 求和被**延迟**到 `ffn_exit.finish()` 返回的 `OwedOutput.reduce_to_dp_local`（`dp_reduce_scatter`），在下一层入口 `complete_owed` 执行。
8. `T_global` 的构成随阶段变：**prefill（SUM_LEN）** = Σ 各卡本地行数（不 padding）；**decode（MAX_LEN）** = `max_len × dp_size`（补 padding 行）。单请求时另一张卡本地行数为 0（`mlp_in_local=(0,2048)`）。
9. `tp>1` 的通信**不是 A2A**：DP gather（复制）+ TP reduce-scatter（求和+切行）。判据：`expert_tokens` 长度 = 全局 expert 数、两卡 MoE 输入相同、输出是部分和。
10. 新 base 上多了同事的 **`ascend_ops`** dispatcher（`--moe-a2a-backend ascend_ops`，prefill 用 `all_to_all_single`、decode 用 MC2），**本 PR 完全没动它**。
11. **`none + ep_size>1` 在 NPU 上从来就不成立**（routing 用全局 expert id，权重按 EP 切）——旧代码会静默算错/崩；新代码在建层时报错。

---

## 10. 未决问题与剩余风险

| # | 问题 | 现状 | 怎么验 |
|---|---|---|---|
| 1 | BF16 非量化档位 | 未验 | 起 BF16 模型，看 `dispatch_out scale=None` |
| 2 | MXFP8（`quant_mode=3`、pair-split scale） | 未验 | 有权重就看 `scale` 是否 3 维 |
| 3 | GGUF + TP>1（finalize 内 all-gather） | 未验 | 有权重就冒烟；看 `combine_out` 宽度 |
| 4 | 其它 MoE 模型（Qwen/GLM/Mixtral/DiT） | 未验 | 共享路径，静态分析风险低；但"融合 shared expert""non-gated"两类值得各冒一次 |
| 5 | `none + ep>1` 的行为变化 | 已改为报错 | 确认团队接受（PR 描述需说明） |
| 6 | `test_resolution_declarations.py` 的冲突 shape | 已改 `ascend_tp→none`，**覆盖语义变弱** | 可选：另找双写者场景补回 |
| 7 | "dtype 未注入 → 静默退回 bf16" | 无兜底断言（用户明确不要） | 遗留项，可后续 PR 加 |
| 8 | DSV4-Flash deepep 冒烟 | 脚本已备，结果未回传 | 见 §8.6 |
| 9 | `ascend_tp` 作为 CLI 值的兼容性 | 已删除 → 老脚本会在参数解析期报错 | PR 描述里写清；若需兼容可加回 alias + warning |

---

## 11. 常见坑（v1 + v2 实战）

1. **必须 `--disable-cuda-graph`**（aclgraph 捕获里打点无效）。
2. **必须 `--disable-radix-cache`**（否则第二次请求命中前缀缓存，`T` 对不上）。
3. **必须 `--skip-server-warmup`**：warmup 会向**每个 DP rank 发一条相同的 prompt**（`http_server.py` 里 `["The capital city of France is"] * num_dp_ranks`），把 `LIMIT` 吃光，你会误以为日志是"自己请求的"。
4. **日志是追加写**，重跑前 `rm -rf <DIR>`；**dump 是同名覆盖**，prefill 会被 decode 覆盖。
5. **判断"这段日志是谁的"**：先看时间戳（prefill 时间 vs curl 时间），再看形状。
6. **`SGLANG_MOE_TRACE_LIMIT` 只限制 trace，不限制 dump**。
7. **不带 `layer=` 的 tag**（`moe_runner` / `pre_permute` / `post_permute`）计数是全局的，很容易被 `LIMIT` 截断——看到它们只出现 2 次，不代表只跑了 2 层。
8. **打点位置必须核对调用点**（见 §6.2）；`transport` 和 `op_comm_*` 都是"写了但不会执行"的典型。
9. **循环 import**：任何被 `quantization/**` 间接 import 的模块（`moe_runner/**`、`hardware_backend/npu/**`）**不能在模块级 import `topk` / `eplb` / `observability` / `disaggregation` / `models`**。否则服务启动即崩（`partially initialized module`）。
10. **改完 git 历史要用 `git show HEAD:<path>` 复核**：只看工作区会出现"我修了但没提交进去"（v2 真实踩过两次）。
11. **强推后要通知用户 `git reset --hard origin/<branch>`**。
12. **不要动用户工作区的未跟踪文件**：`DTS/`、`deepseek_v4_pro_single_tp16.sh` 都是他的笔记/脚本，清理仓库时别顺手删。
13. **topk 格式**：新 permute 只接受 `StandardTopKOutput`；`BypassedTopKOutput` / `TritonKernelTopKOutput` 会显式 `NotImplementedError`。将来若要让 none 路径支持 fp4 trtllm，需要在这里扩展，而不是放行。

---

## 12. 验收结论（可直接引用的措辞）

在 Ascend NPU 上以 DeepSeek-V2-Lite-W8A8 验证 `feat/moe-none-merge` 相对基线 `feat/moe-trace`：

| 配置 | 路径 | 覆盖层 | 结果 |
|---|---|---|---|
| tp=1 | prefill（9 token → 54 行） | L1/L13/L26 | **逐比特一致**（md5 相同） |
| tp=1 | decode（1 token → 6 行） | L1/L13/L26 | **逐比特一致**（md5 相同） |
| tp=2 + dp-attention | prefill（全局 9 token，两卡各一份 dump） | L1/L13/L26 | **两卡 × 三层 × 四个张量全部逐比特一致** |

比对张量：`dispatch_out`（hs / scale / row_idx / expert_tokens / topk_ids / topk_weights）、`runner_out`、`combine_out`、`forward_impl_out`。

另外：

- `test/registered/unit/server_args/test_resolution_declarations.py`：**14 passed**；
- 服务冒烟：单卡起服务 + 一条 chat 请求，输出通顺中文；
- **两卡的 `dispatch_out` 逐位相同**（同一全局 batch 上冗余计算 routing）→ 实证了"确定性"这个 TP 正确性的隐含前提。

**未验证（必须在 PR 里如实写）**：tp>1 的 decode、BF16/MXFP8/GGUF、其它模型、DSV4 冒烟。

---

## 13. 建议的开场（新窗口怎么接）

1. 先问用户"这次想推进哪一项"：**补覆盖**（§10 的 1/2/3/8）、**跟 PR 评审**、还是**继续深挖某段代码**。
2. 若是补覆盖：一次只给一个冒烟，明确"看哪一行日志就能判定成功"。
3. 若是跟 PR：先让他贴 PR 的 review 意见或 CI 结果，**不要凭猜测回复 reviewer**。
4. 任何新结论都要落到"日志行 / dump / 单测"三者之一；**§9 的事实可以直接用，不要重新推导**。
5. 用户偏好结构化解释（"为什么/在哪一层/和别处怎么配合"），并且非常重视**证据链**——他曾经用一个 `(6,2048)` 的形状差异，直接推翻了 v1 文档里的两条结论。

---

## 14. 附录 A：一页速查（"想找什么 → 去哪看"，已更新到新分支）

| 想找什么 | 去哪看 |
|---|---|
| 参数默认值 / 可选值 | `arg_groups/fields/exec_.py`（`moe_a2a_backend`）+ `fields/spec.py`（draft） |
| 参数归一化（旧 `ascend_tp→none`） | ❌ 已删除；现在 CLI 直接拒绝 `ascend_tp` |
| 运行时 backend 枚举 | `layers/moe/utils.py`（`MoeA2ABackend`） |
| **dispatcher 工厂（总入口）** | `layers/moe/fused_moe_triton/layer.py::create_moe_dispatcher` |
| **routing 状态 + 内核工厂** | `hardware_backend/npu/moe/dispatch.py`（`get_routing` / `_QUANT_MODE`） |
| **两个 permute 钩子** | `layers/moe/moe_runner/ascend.py`（`pre_permute_standard_to_ascend` / `post_permute_ascend_to_standard`） |
| dtype 的 8 个声明点 | `hardware_backend/npu/quantization/moe_methods.py::_set_moe_dispatch_dtype` 的调用点 |
| GGUF 的 finalize all-gather | `layers/quantization/gguf.py`（打标记）+ `dispatch.py::get_routing`（包 wrapper） |
| CANN routing/finalize 算子封装 | `hardware_backend/npu/moe/init_routing.py` / `finalize_routing.py` |
| **MoE 段的 DP gather / 切行** | `layers/layer_boundary/`（`ffn_boundary.prepare` / `complete_now`）——**不是** `moe_transport.py`（V4 专用） |
| MoE 输出后的 TP 求和 | `OwedOutput.reduce_to_dp_local` → `dp_reduce_scatter`（延迟到下一层入口） |
| permute 池查表 | `layers/moe/moe_runner/runner.py` + `base.py::PermuteMethodPool` |
| 并行度布局 | `runtime_context.py::parallel_widths_of`（`moe_tp_size = tp // ep // moe_dp`） |
| 另一个 dispatcher（同事的） | `token_dispatcher/ascend_ops.py`（真 A2A，本 PR 未动） |

---

## 15. 附录 B：验收分级（L0–L3）

| 级别 | 判据 | 我们的状态 |
|---|---|---|
| **L0** | 服务能起、不 assert、请求能返回 | ✅ |
| **L1** | 模块级 `mlp.experts` 输出逐元素 `max_abs_diff == 0` | ✅（用 dump 等价实现了更强的 L3） |
| **L2** | greedy 解码 N 步逐字符一致（`temperature=0`） | ✅（8 步冒烟一致；未做 32 步正式 A/B） |
| **L3** | MoE 内部 dump 逐张量一致（`hs` / `scale` / `row_idx` / `expert_tokens` / `topk_*`） | ✅ **prefill + decode + tp=2 均达到** |

> 最省事的 L2 判据：两条分支用同一条 chat 请求（`temperature=0, max_tokens=32`），比较返回文本是否**逐字符相同**。

---

## 16. 附录 C：交付 / PR 描述要素（来自旧任务书，仍然适用）

1. 改了哪些文件、每个文件承担什么职责；
2. `ascend_tp` / 默认 `none` / 显式 `none` 的**最终调用链**；
3. 哪些 Ascend 专用逻辑被**复用 / 移动 / 删除**，为什么；
4. 测试命令、环境、结果；
5. 是否完成真实 NPU 端到端验证；
6. **尚未验证**的硬件 / 模型 / 量化 / EP 组合（照实写，别用"逻辑已统一"概括）；
7. 需要后续独立处理的工作：`ascend-ops`（同事在做）、DeepEP-Ascend、动态 EPLB、W4（runner 内 a2a 条件分支数据驱动）。

---

## 17. 附录 D：与旧文档的关系（已合并删除）

2026-10-23 已把以下三份旧文档的**有效内容**吸收进本文，并删除原文件：

| 旧文件 | 吸收了什么 | 丢弃了什么 |
|---|---|---|
| `moe_a2a_ascend_tp_to_none_task_guide.md` | 原始验收标准、四条设计方向、三层概念、布局公式、DeepEP 强制 ep=tp、交付清单 | 旧的分支/行号、以及"NPU none+ep>1 可用"这个错误假设 |
| `ascend_tp_to_none_analysis.md` | 三层概念表、W1–W6 实施顺序、7 条风险清单、验证矩阵、四个入口 | 旧的 `ascend_tp` 兼容别名方案（我们选择直接删除） |
| `moe_learning_route.md` | 零改码模块级 dump、L0–L3 验收分级、一页速查、形状预期表 | `expert_tokens` 是"累加和"、`moe_transport.py` 是 V2 的 gather 位置（均已由 §9 修正） |

**唯一一处旧文档对了、而我们没做的**：`ascend_tp_to_none_analysis.md` §3.5 建议"把 runner 里 `is_deepep()` 的激活选择改成数据驱动（`pertoken_scale is None`）"。我们**保留了** `is_deepep() or is_ascend_ops()` 的显式判断，行为等价但仍是分支。这是 W4，列为后续项。
