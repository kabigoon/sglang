# 修复记录：PR #50205 代码 + 新代 NPU 镜像 → 让 server 能起

> 一句话：**新镜像不再打包老算子包，PR 代码还在用老算子**。补装老算子包即可，
> 不需要改 PR 代码。全过程在一个容器里完成，约 15 分钟（不含下载/传输）。

## 1. 问题现象

起 `run_dsv41.sh`，权重加载成功、DSpark draft 挂上、KV 池分配完成，然后崩在
**decode 图捕获的第一个真实 forward**：

```
File ".../hardware_backend/npu/dsv4/dsv4_rope.py", line 163, in apply_rotary_mul_inplace
    torch.ops.custom.inplace_partial_rotary_mul(
AttributeError: '_OpNamespace' 'custom' object has no attribute 'inplace_partial_rotary_mul'
```

## 2. 根因

| 事实 | 证据 |
|---|---|
| PR 代码用 **13 个** `torch.ops.custom.*` | `grep -rhoE 'torch\.ops\.custom\.[a-z_]+' python/sglang/srt \| sort -u` |
| 这些算子由独立包 `custom_ops` + CANN vendor `customize`/`custom_transformer` 提供 | PR 自带 `docker/npu.Dockerfile` |
| 从 `sgl-kernel-npu` **2026.11.0.post4** 起，这两个发行物被**取消**，vendor 内联进 wheel，改注册到 `torch.ops.npu.*` + `sgl_kernel_npu.operators.*` | release 资产：post1–post3 有 `custom-ops-*.zip`，post4 起消失 |
| 官方同步提交 #50203（`9ddaa93234`，11-14）把主干这 18 处迁到了新 API，**但不在 PR 分支里** | `git merge-base --is-ancestor 9ddaa93234 pr-50205` → 否 |

所以：**镜像换了一代算子注册方式，PR 分支还是老那一代**。两种修法——补老包（本次采用）
或迁代码到新 API。

## 3. 修复步骤（容器内）

```bash
cd /home/a00821909/sglang
git pull --ff-only            # 拿到 _tmp_preflight/ 里的脚本

# ① 两个包放进 _tmp_preflight/zips/（从 Windows/内网传进来即可，该目录已 gitignore）
#    custom-ops-2026.11.0.post3-torch2.13.0-cann9.2.0-a3-aarch64.zip          73646297 B
#    ops-transformer-2026.11.0.post3-torch2.13.0-cann9.2.0-a3-aarch64.zip     12046381 B

# ② 安装（自动探测 torch/cann/arch；先查 size 再查 sha256，任一不符即中止）
LOCAL_DIR=$PWD/_tmp_preflight/zips bash _tmp_preflight/fix_install_custom_ops.sh

# ③ 复验注册层
bash _tmp_preflight/check_image.sh     # 期望 OPS_CUSTOM=13/13  IDEAL=YES
bash _tmp_preflight/check_import.sh    # 期望 OPS_AFTER_INIT=10/10  IMPORT_FAILS=0

# ④ 起服务（run_dsv41.sh 顶部本来就 source vendor，见下）
bash run_dsv41.sh

# ⑤ 固化容器（宿主机执行）
docker ps --format '{{.ID}}  {{.Names}}'
docker commit <容器ID> sglang-npu-a3:pr50205-customops
```

安装脚本实际做两件事（对应 PR 的 Dockerfile）：

```bash
./CANN-custom_ops-none-linux.aarch64.run   --install-path=/usr/local/Ascend/cann-9.2.0/opp
./cann-ops-transformer-custom_linux.aarch64.run --install-path=/usr/local/Ascend/cann-9.2.0/opp
python3 -m pip install ./custom_ops-1.0-cp312-cp312-linux_aarch64.whl
```

## 4. **启动脚本必须 source** 这两行

```bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
```

不 source 的话 `import custom_ops` 会 `ImportError`（加载不到 aclnn 库），
`init_npu_backend()` 里被 try/except 吞成 warning，然后 forward 时又回到那个 `AttributeError`。

## 5. 依赖的算子包（出处 + 校验）

官方仓库 `github.com/sgl-project/sgl-kernel-npu` 的 release，资产由 `github-actions[bot]` 发布。
sha256 取自 GitHub `assets.digest`（不是本地计算）：

| tag | 文件 | size | sha256 |
|---|---|---|---|
| post3 | custom-ops | 73646297 | `827d70c9b31f4baf1c3ad08bb5f1d7ef0719ce04abd2edf1d5f0f1eb685fe048` |
| post3 | ops-transformer | 12046381 | `b65172505dd9933501de421fbca5e9fa1b61bb6598c9cfb4eb3476fac15af85b` |
| post1 | custom-ops | 73645995 | `e10ca2daeb3a9fb48d97c329594a0f8c6299b79b917ad0ba3382e2782cb89196` |
| post1 | ops-transformer | 12044483 | `0f0e1a8cc6803b847e7759f50fa9789832ca6398baf44a835972dfe96ccd2dbb` |

直链模板：
`https://github.com/sgl-project/sgl-kernel-npu/releases/download/<TAG>/{custom-ops,ops-transformer}-<TAG>-torch2.13.0-cann9.2.0-a3-aarch64.zip`

## 6. 踩过的坑（都已在脚本里处理）

| 坑 | 表现 | 处理 |
|---|---|---|
| 公司代理做 SSL 拦截 | `curl (60) self-signed certificate` / `wget 退出码=5` | 四级阶梯：默认CA → 系统CA → `-k` → `wget --no-check-certificate`，并用 sha256 兜底 |
| 代理慢/断流 | 下到一半失败 | `curl -C -` 断点续传 + `<1KB/s 持续 90s` 判死重试 |
| 下载失败无信息 | 只有一句 FAIL | 打印 proxy、curl/wget 路径、HTTP 码、退出码 |
| `import custom_ops` 顺序 | 先 import 它 → `ImportError`；先 torch/torch_npu → OK | 探针按 torch → torch_npu → custom_ops 顺序 |
| 未 source 导致假阴性 | 装好了但 `0/13` | `check_image.sh` 自己 source | 
| Windows 传输后改名 | `LOCAL_DIR` 里 glob 匹配不到 | 用 `custom-ops*` / `ops-transformer*` 通配 |

## 7. 尚未验证 / 回退方案

- 当前是 **post3 老算子 + post4 新 kernel 包** 的混搭，官方未测试该组合；
  注册层已验证齐（13/13、`OPS_AFTER_INIT=10/10`），**调用契约未验证**。
- 若起服务报 `TypeError` / `RuntimeError: ... schema/expects ...`（不是 `AttributeError`）：
  1. `pip uninstall -y custom_ops` 后 `TAG=2026.11.0.post1` 重装（更贴近 PR 作者钉的版本）；
  2. 仍不行则整包把 `sgl_kernel_npu` 也降到 post3，消除混搭；
  3. 或把主干 #50203 的 18 处迁移移植到 PR 分支（改代码，最干净）。
