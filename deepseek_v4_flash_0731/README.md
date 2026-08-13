# DeepSeek-V4-Flash-0731：8 机全参数 SFT

本目录整理自 `train/script/0805_dpsk` 的最终可运行版本，只保留训练、启动、
环境准备脚本和实际运行所需的补丁；约 26 MB 的实验日志未纳入仓库。

## 目录结构

```text
deepseek_v4_flash_0731/
├── launch_dpsk_dsw24_31.sh  # 8 机 SSH/tmux launcher
├── prepare_dpsk_env.sh      # 创建专用 venv、固定代码版本并应用补丁
├── train_dpsk.sh            # 单节点训练入口
└── patches/                 # 最终运行环境相对上游 commit 的改动
```

## 已验证环境与代码版本

以下版本来自实际运行的 `swift_dsv4_0805_updated` 环境：

| 组件 | 版本或提交 |
| --- | --- |
| Python | 3.12.12 |
| PyTorch | 2.10.0+cu130 |
| NCCL | 2.28.9 |
| Transformers | 5.14.1 |
| Transformer Engine | 2.14.1，commit `366798ef8a0a00d8f2c1650d11e7e623d7c33e26` |
| FlashAttention | 2.8.3 |
| Megatron-Core | commit `fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1` |
| mcore-bridge | commit `60a0d696d4e95b1c2f3dd560b3109e04dc799c4a` |
| ms-swift | commit `9298fb8a970aa0d07e362de02aace171cc5acdf5` |

`prepare_dpsk_env.sh` 已固定后三个代码仓库的提交和 Transformers 版本。
PyTorch、Transformer Engine、FlashAttention 与 NCCL 由 `BASE_PYTHON` 所在基础
环境提供，需按目标机器的 CUDA 驱动单独安装。

## 本地补丁

实际跑通环境包含三组相对上述提交的补丁，环境脚本会自动应用：

- `megatron-core.patch`：保证 optimizer shard 是 leaf tensor，并对 unfused CSA
  的 gather/FP32 临时空间按行分块，避免 64K 上下文产生约 20 GiB 的单次分配。
- `mcore-bridge.patch` 与 `ms-swift.patch`：禁止把 DeepSeek-V4 的 DSpark
  speculative-decoding block 当作 Megatron generic MTP 训练。

补丁会从 venv 的 `site-packages` 目录以 `patch -p1` 应用；脚本可重复执行，并会
识别已经应用的补丁。

## 准备环境

原实验以已有的 CUDA/PyTorch 环境作为基础，创建独立的
`--system-site-packages` venv：

```bash
cd /share/project/chaofan/code/self_evolving_v15

BASE_PYTHON=/share/project/chaofan/envs/swift_0426/bin/python \
SWIFT_ENV=/share/project/chaofan/envs/swift_dsv4_0805_updated \
MODEL_PATH=/share/project/shared_models/DeepSeek-V4-Flash-0731 \
bash train/smoke/deepseek_v4_flash_0731/prepare_dpsk_env.sh
```

环境脚本最后会确认模型能解析为 `deepseek_v4`，并检查 Megatron、mcore-bridge、
ms-swift 均可导入。

## 默认训练配置

原实验使用 dsw-24 到 dsw-31 共 8 台机器、每台 8 张 H100 80GB。最终脚本默认值
如下（早期实验中的 PP=2/MTP=1 已不再使用）：

```text
TP=1, PP=16, EP=4, ETP=1, CP=4, dense DP=1, MTP=0
pipeline layout=Et*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*2|t*2|t*2|t*2|t*2L
max_length=65,536
micro_batch_size=1
global_batch_size=64
num_train_epochs=3
LR=1e-6, min LR=1e-7
```

此外启用 FP8 blockwise、完整 CPU optimizer offload、FP32 main gradients、full
recompute、packing 和 contiguous context partition。当前 Hybrid Attention 要求
TP=1；DSpark 不兼容 generic MTP，因此 `MTP_NUM_LAYERS` 默认必须为 0。

## 启动前检查

脚本默认数据路径来自原实验环境，可通过 `DATASET_FILES`、`VAL_DATASET`、
`MODEL_PATH`、`SWIFT_ENV` 和 `SSH_CONFIG` 覆盖。

只检查本机路径，不连接 SSH：

```bash
bash train/smoke/deepseek_v4_flash_0731/launch_dpsk_dsw24_31.sh --dry-run
```

对 8 台机器做 SSH、tmux、训练进程、环境和模型检查，但不启动训练：

```bash
bash train/smoke/deepseek_v4_flash_0731/launch_dpsk_dsw24_31.sh --preflight-only
```

launcher 不会终止已有进程；检测到训练进程或同名 tmux session 时默认退出。

## 启动训练

所有节点共享代码、模型、环境、数据和输出路径时，在 launcher 所在机器执行：

```bash
bash train/smoke/deepseek_v4_flash_0731/launch_dpsk_dsw24_31.sh
```

若需要让 launcher 先准备共享环境：

```bash
bash train/smoke/deepseek_v4_flash_0731/launch_dpsk_dsw24_31.sh --prepare-env
```

使用一条小数据做单 step smoke test：

```bash
DATASET_FILES=/shared/data/smoke.jsonl \
VAL_DATASET= \
TRAIN_ITERS=1 \
RUN_TAG=dpsk_v4_flash_0731_smoke \
SESSION_NAME=train_dpsk_v4_flash_0731_smoke \
bash train/smoke/deepseek_v4_flash_0731/launch_dpsk_dsw24_31.sh
```

如果仓库不位于原来的 `train/smoke` 层级，可显式设置 `V15_ROOT`，用于解析默认
数据和输出路径。网卡、IB HCA、缓存、并行策略及数据加载参数均可通过脚本中同名
环境变量覆盖。
