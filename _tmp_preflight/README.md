# _tmp_preflight —— 临时目录（用完即删）

> 用途：在**不起服务**的前提下，判断这台机器/这个镜像能不能跑 PR #50205 这份代码。
> 归属：`pr-50205-dsv41-a3-w8a8` 分支上的临时文件，**不属于 PR 内容**，用完执行
> `git rm -r _tmp_preflight && git commit -m "chore: drop preflight"` 即可清掉。
> 全部只读：不加载权重、不建 KV 池、不抓图、不改任何文件（日志只写 `/tmp`）。

## 为什么需要它

PR 分支的代码还在用老一代算子注册方式（`torch.ops.custom.*`，13 个），
而新镜像从 `sgl-kernel-npu` 的 `2026.11.0.post4` 起**不再打包** `custom-ops` /
`ops-transformer`，于是 `torch.ops.custom.*` 全空，第一次 forward 就会
`AttributeError: '_OpNamespace' 'custom' object has no attribute 'inplace_partial_rotary_mul'`。

## 三个脚本

| 脚本 | 做什么 | 耗时 | 需要 NPU 卡 |
|---|---|---|---|
| `check_image.sh` | 镜像代际（pip / custom_ops / opp vendors）+ 13 个算子逐个探测 + 结论 | ~10 s | 否 |
| `dump_ops.sh` | 把 `custom` / `npu` / `cann_ops_transformer` 三个命名空间已注册算子全量导出（部分命中时才需要） | ~10 s | 否 |
| `check_import.sh` | 用本仓库代码干跑 import（抓 H15：跑的不是这份代码 / 缺 python 依赖） | ~30-60 s | 否 |

## 怎么跑

**推荐：进容器里跑**（和你平时启动的方式一致）

```bash
cd /home/a00821909/sglang
bash _tmp_preflight/check_image.sh
```

**或者：从宿主机直接丢进容器跑**（仓库按只读挂到 `/repo`）

```bash
docker run --rm -v /home/a00821909/sglang:/repo:ro \
  --entrypoint bash <镜像名或ID> /repo/_tmp_preflight/check_image.sh
```

按需再跑另外两个：

```bash
bash _tmp_preflight/dump_ops.sh      > /tmp/ops_dump.txt    # 需要看具体缺哪些时
bash _tmp_preflight/check_import.sh                          # 排查 import 问题
```

## 每个脚本跑完末尾都会打印一个「回传块」

git 通道是单向的，所以你只要**复制末尾 `===== 回传块 =====` 里的那几行**发回来即可，
不用贴整份日志（完整日志会落到 `/tmp/dsv41_preflight/`，路径也会打印出来）。

## 判据表（拿到回传块之后怎么对）

| 回传块特征 | 含义 | 下一步 |
|---|---|---|
| `CUSTOM_OPS_IMPORT=OK` 且 `OPS_CUSTOM=13/13` | 老代镜像，与 PR 同代 | 直接起服务，不再有 rope 报错 |
| `CUSTOM_OPS_IMPORT=FAIL(...)` 且 `OPERATORS_ROPE=OK` | **新代镜像，与 PR 代码不匹配** | 三选一：换老镜像 / 补装 post3 的 custom-ops / 移植 #50203 |
| `OPS_CUSTOM` 介于 1/13 和 12/13 之间 | 镜像介于两代之间（vendor 装了一半） | 跑 `dump_ops.sh`，我按 A3 可达性逐条判断 |
| `TORCH=FAIL` 或 `OPERATORS_ROPE=FAIL` 且 custom 也 FAIL | 镜像本身不对（不是 NPU 运行时） | 先确认镜像 |

## 注意

- 脚本不设 `set -e`：某一项失败会继续跑，方便一次拿到全貌。
- 不依赖 `jq`/`gh`，只用 bash + python3 + grep。
- `check_import.sh` 只 import，不实例化模型，所以不需要权重路径。
