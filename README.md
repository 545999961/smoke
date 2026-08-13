# 训练复现实验

- Qwen3.5-122B-A10B 长上下文训练复现：见本页。
- [DeepSeek-V4-Flash-0731 8 机全参数 SFT](deepseek_v4_flash_0731/README.md)：
  包含最终训练脚本、固定依赖提交和实际运行环境使用的补丁。

---

# Qwen3.5-122B-A10B 训练复现

`train.sh 64k` 是目前脚本的主要配置

## 环境需求

原训练环境的主要版本如下：

- Python 3.12.12
- PyTorch 2.10.0 + CUDA 13.0
- transformers 5.2.0
- Transformer Engine 2.14.1
- FlashAttention 2.8.3
- mcore-bridge 1.2.1
- ms-swift commit `27ed3cadebefff639b2ad7e7488ceabe0f4496cb`
- Megatron-LM commit `f2dcd421b2addb98360b7c751e721b02ba4f2955`

PyTorch 应根据目标机器的 CUDA 驱动单独安装。其余代码可以直接 clone：

```bash
git clone https://github.com/modelscope/ms-swift.git
git -C ms-swift checkout 27ed3cadebefff639b2ad7e7488ceabe0f4496cb
pip install -e ms-swift

git clone https://github.com/NVIDIA/Megatron-LM.git
git -C Megatron-LM checkout f2dcd421b2addb98360b7c751e721b02ba4f2955
pip install -e Megatron-LM

pip install \
  transformers==5.2.0 \
  mcore-bridge==1.2.1 \
  transformer_engine[pytorch]==2.14.1
MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation
```

## 训练资源需求

原实验使用 8 个节点、每节点 8 张 GPU，共 64 张 GPU。默认并行配置为：

```text
TP=8, PP=4, EP=8, ETP=1, CP=1
```

## 两档数据

### 64K

`data/long_64k_train.jsonl` 包含 3 条 60k 左右的多轮工具调用数据。

`train.sh 64k` 使用：

```text
max_length=65,536
CP=1
global_batch_size=128
num_train_epochs=2
```

### 200K+

`data/long_256k_train.jsonl` 包含 2 条 200k 左右的多轮工具调用数据。

`train.sh 200k` 使用：

```text
max_length=262,144
CP=2
global_batch_size=1
num_train_epochs=1
```

## 本地构造重复数据

`repeat_jsonl.py` 按顺序循环读取输入 JSONL，直到生成指定行数。输入文件、
输出文件和目标行数依次作为三个参数传入。

构造 profiling 数据：

```bash
# 3 条 64K 唯一数据循环到 128 行，对齐默认 global batch 128。
python3 repeat_jsonl.py \
  data/long_64k_train.jsonl \
  generated_data/long_64k_repeat128.jsonl \
  128

# 2 条 200K+ 唯一数据循环 16 次，共生成 32 行。
python3 repeat_jsonl.py \
  data/long_256k_train.jsonl \
  generated_data/long_256k_repeat32.jsonl \
  32

wc -l generated_data/*.jsonl
```

使用生成后的数据：

```bash
TRAIN_DATASET=generated_data/long_64k_repeat128.jsonl \
bash train.sh 64k

TRAIN_DATASET=generated_data/long_256k_repeat32.jsonl \
bash train.sh 200k
```

64K 数据生成 128 行后，每个 epoch 至少覆盖 1 个完整 global batch，默认 2 epochs 共运行 2 个 step。200K 数据生成 32 行后，默认 global batch 为 1， 每个 epoch 运行 32 个长上下文 step。

## Dry-run

dry-run 只检查参数和并行配置，不启动训练：

```bash
DRY_RUN=1 bash train.sh 64k
DRY_RUN=1 bash train.sh 200k
```

## 多机启动

每个节点执行同一个脚本，所有参数保持一致，只修改 `NODE_RANK`。

rank 0 示例：

```bash
MASTER_ADDR=10.0.0.1 \
MASTER_PORT=29500 \
NNODES=8 \
NPROC_PER_NODE=8 \
NODE_RANK=0 \
MODEL_PATH=/shared/model \
TRAIN_DATASET=/shared/data/long_64k_repeat128.jsonl \
OUTPUT_DIR=/shared/output \
CACHE_ROOT=/shared/cache \
bash train.sh 64k
```

其余节点将 `NODE_RANK` 设置为 1～7。`MASTER_ADDR` 必须是所有节点均可访问的
rank-0 地址。

200K 训练只需把最后一行改为：

```bash
TRAIN_DATASET=/shared/data/long_256k_repeat32.jsonl \
bash train.sh 200k
```

若网卡不是 `eth0`，需要覆盖：

```bash
NCCL_SOCKET_IFNAME=your_nic \
GLOO_SOCKET_IFNAME=your_nic \
bash train.sh 200k
```

需要限制 IB HCA 时可额外设置 `NCCL_IB_HCA`。

## 使用完整训练数据

使用完整数据时创建一个文本清单，每行放一个 JSONL 路径，空行和 `#` 注释会
被忽略：

```text
/shared/data/train_1.jsonl
/shared/data/train_2.jsonl
```

然后在所有节点增加：

```bash
DATASET_MANIFEST=/shared/data/datasets.txt \
VAL_DATASET=/shared/data/valid.jsonl \
bash train.sh 64k
```
