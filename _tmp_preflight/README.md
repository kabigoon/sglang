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
| `fix_install_custom_ops.sh` | 给**新代镜像**补装 post3 的 `custom-ops` / `ops-transformer`（只在 `VERDICT=NEW_GEN_MISMATCH` 时用） | 1-3 min | 否 |
| `find_artifacts.sh` | 无外网时在本机翻找现成的 zip / wheel / vendor 目录 | ~1 min | 否 |
| `check_any_image.sh` | 从宿主机一条命令检查任意镜像（内部就是 `docker run ... check_image.sh`） | ~15 s | 否 |

## 怎么跑

**推荐：进容器里跑**（和你平时启动的方式一致）

```bash
cd /home/a00821909/sglang
bash _tmp_preflight/check_image.sh
```

**或者：直接在宿主机上检查某个镜像**（不用先进容器）

```bash
bash _tmp_preflight/check_any_image.sh <镜像名或ID>
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

## 路线 B：现场补装（仅当 `VERDICT=NEW_GEN_MISMATCH`）

```bash
bash _tmp_preflight/fix_install_custom_ops.sh          # 会自己探测 torch/cann/arch
bash _tmp_preflight/check_image.sh                     # 复验，期望 OPS_CUSTOM=13/13
```

- 无外网时：先把 `custom-ops-2026.11.0.post3-torch2.13.0-cann9.2.0-a3-aarch64.zip` 和
  同名 `ops-transformer-*.zip` 拷进机器，然后
  `LOCAL_DIR=/path/to/zips bash _tmp_preflight/fix_install_custom_ops.sh`；
- 装完**必须**让启动脚本 source 这两行（你现在的脚本里已经有了，之前因为目录不存在而静默失败）：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
```

- 回滚：`pip uninstall -y custom_ops`，再删掉 `opp/vendors/{customize,custom_transformer}`；
- 如果容器是一次性的，装成功后建议 `docker commit <容器> sglang-npu-a3:pr50205-post3ops` 固化下来。

## 下载失败怎么办

两个包分别是 **70 MB**（custom-ops）和 **11 MB**（ops-transformer）。

1. **先重跑 `fix_install_custom_ops.sh`**：新版会打印 `proxy=` / `工具:` /
   `github.com -> HTTP xxx` / `资产URL -> HTTP xxx redirects=n`，以及 curl、wget 各自的退出码。
2. **看有没有现成的**：`bash _tmp_preflight/find_artifacts.sh`
   （搜 `/home /root /tmp /mnt /data /opt /workspace`，包括 pip 缓存和老镜像留下的包）。
3. 都找不到再传：在有外网的机器下载这两个 zip，拷进 NPU 机器，然后
   `LOCAL_DIR=/path/to/zips bash _tmp_preflight/fix_install_custom_ops.sh`。

| 自检看到 | 含义 |
|---|---|
| `HTTP 000` / `timed out` | 出不去，需要 `export https_proxy=...` |
| `HTTP 302` 之后再卡住 | `objects.githubusercontent.com` 被挡（企业网常见；git 能通不代表 release 能下） |
| curl 退出码 `60` / `wget` 报 `Self-signed certificate` | **公司代理在做 SSL 拦截**（如华为 `Huawei Web Secure Internet Gateway CA`）。脚本会自动走四级阶梯：默认CA → `--cacert 系统CA` → `curl -k` → `wget --no-check-certificate`，并在最后用 **sha256** 校验，所以 `-k` 也不会装进被篡改的包 |
| `HTTP 200` 但下载中断 | 带宽/超时，脚本已带 `--retry 5`，可重跑（wget 分支支持 `-c` 续传） |

### 已知 good 校验和（post3 / torch2.13.0 / cann9.2.0 / a3 / aarch64）

```
827d70c9b31f4baf1c3ad08bb5f1d7ef0719ce04abd2edf1d5f0f1eb685fe048  custom-ops-2026.11.0.post3-torch2.13.0-cann9.2.0-a3-aarch64.zip  (73646297 B)
b65172505dd9933501de421fbca5e9fa1b61bb6598c9cfb4eb3476fac15af85b  ops-transformer-2026.11.0.post3-torch2.13.0-cann9.2.0-a3-aarch64.zip  (12046381 B)
```

手动校验：`sha256sum <file>` 对比上面两行。**校验不过就别装**——可能是代理返回了错误页。

## 注意

- 脚本不设 `set -e`：某一项失败会继续跑，方便一次拿到全貌。
- 不依赖 `jq`/`gh`，只用 bash + python3 + grep。
- `check_import.sh` 只 import，不实例化模型，所以不需要权重路径。
