#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V15_ROOT="${V15_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RUN_TAG="${RUN_TAG:-dpsk_v4_flash_0731_full_sft}"

MODEL_PATH="${MODEL_PATH:-/share/project/shared_models/DeepSeek-V4-Flash-0731}"
SWIFT_ENV="${SWIFT_ENV:-/share/project/chaofan/envs/swift_dsv4_0805_updated}"
SFT_DATA_DIR="${SFT_DATA_DIR:-/share/project/kunluo/Projects/ScienceAgent/GeneralSearchAgent/SearchAgent-SLM/dataset/SFT}"
CLAUDE_BC_DATA_DIR="${CLAUDE_BC_DATA_DIR:-/share/project/chaofan/code/self_evolving/unify_browsecomp/scripts/0514/keyturn_loss_eval_data}"
KEYTURN_MULTILOSS_ROOT="${KEYTURN_MULTILOSS_ROOT:-${V15_ROOT}/data/source/previous-trajectories/train_data_key_turn_multiloss/previous-sft-122b_bc}"
VAL_DATASET="${VAL_DATASET-/share/project/chaofan/code/self_evolving/unify_browsecomp/results/collect/ws_processed/0501/validation/valid.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${V15_ROOT}/train/result/0805_dpsk/${RUN_TAG}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/runs}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/${RUN_TAG}}"
CACHE_ROOT_BASE="${CACHE_ROOT_BASE:-/share/project/chaofan/cache/modelscope/0805_dpsk/${RUN_TAG}}"
NCCL_SHARED_LIB_DIR="${NCCL_SHARED_LIB_DIR:-/share/project/kunluo/Libs/nccl-2.28.3/lib}"

NODE_RANK="${NODE_RANK:-${RANK:-${SLURM_NODEID:-0}}}"
NNODES="${NNODES:-${NUM_NODES:-${SLURM_NNODES:-8}}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR must be set by the launcher or scheduler}"
MASTER_PORT="${MASTER_PORT:-29845}"

# DeepSeek-V4's current Hybrid Attention implementation requires TP=1.
# DP=1: PP=16 puts only two or three decoder layers on each 80-GiB GPU.
# With EP=4, the expert parallel group is ETP*EP*PP=64, so each expert and
# dense data-parallel group spans the full 64-GPU job.
TP="${TP:-1}"
PP="${PP:-16}"
EP="${EP:-4}"
ETP="${ETP:-1}"
CP="${CP:-4}"
PIPELINE_MODEL_PARALLEL_LAYOUT="${PIPELINE_MODEL_PARALLEL_LAYOUT:-Et*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*2|t*2|t*2|t*2|t*2L}"
# DeepSeek-V4 DSpark is unsupported by Megatron's generic MTP layer. Keep
# this at zero for SFT; the new environment rejects nonzero values safely.
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-0}"
# Megatron only permits BF16 main gradients with its precision-aware optimizer.
# That optimizer is incompatible with CPU offload for these sharded DeepSeek-V4
# weights, so use FP32 gradients and rely on PP=16 plus full CPU offload for
# the DP=1 memory budget.
MAIN_GRADS_DTYPE="${MAIN_GRADS_DTYPE:-fp32}"

TRAIN_TYPE="${TRAIN_TYPE:-full}"
MAX_LENGTH="${MAX_LENGTH:-65536}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
# Set TRAIN_ITERS for a bounded smoke run. When it is nonempty it takes
# precedence over epochs, which is necessary when a tiny dataset is smaller
# than one global batch.
TRAIN_ITERS="${TRAIN_ITERS:-}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
LR="${LR:-1e-6}"
MIN_LR="${MIN_LR:-1e-7}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
LOSS_SCALE="${LOSS_SCALE:-default+ignore_empty_think}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-16}"
# Bound the reference CSA sparse-attention workspace at 64K. The fused
# FlashMLA path is unavailable in this environment, so this preserves its
# FP32 math while avoiding a single ~20-GiB gathered-KV allocation.
DSV4_UNFUSED_ATTN_CHUNK_ROWS="${DSV4_UNFUSED_ATTN_CHUNK_ROWS:-256}"
MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-1e-3}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-64}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-1}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-false}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-false}"
PACKING="${PACKING:-true}"
PACKING_LENGTH="${PACKING_LENGTH:-${MAX_LENGTH}}"
PACKING_NUM_PROC="${PACKING_NUM_PROC:-1}"
GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-false}"
SEQUENCE_PACKING_SCHEDULER="${SEQUENCE_PACKING_SCHEDULER:-dp_balanced}"
CP_PARTITION_MODE="${CP_PARTITION_MODE:-contiguous}"
MANUAL_GC="${MANUAL_GC:-true}"
MANUAL_GC_STEPS="${MANUAL_GC_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-200}"
EVAL_STEPS="${EVAL_STEPS:-200}"
EVAL_ITERS="${EVAL_ITERS:--1}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
DRY_RUN="${DRY_RUN:-0}"

DEFAULT_DATASETS=(
  "${SFT_DATA_DIR}/browsecomp_full.jsonl"
  "${CLAUDE_BC_DATA_DIR}/claude_browsecomp_kstep_non_react_loss_eval.jsonl"
  "${CLAUDE_BC_DATA_DIR}/claude_browsecomp_summary_all_non_react_loss_eval.jsonl"
  "${CLAUDE_BC_DATA_DIR}/claude_browsecomp_kstep_react_loss_eval.jsonl"
  "${CLAUDE_BC_DATA_DIR}/claude_browsecomp_summary_all_react_loss_eval.jsonl"
  "${SFT_DATA_DIR}/processed_64k_hle_w_tool_filter_train.jsonl"
  "${SFT_DATA_DIR}/PaperDeepSearch--Model-Qwen35-122B-SFT--Direct--PassAt1-Range0to2000--0614--DefaultSystemPrompt.jsonl"
  "${SFT_DATA_DIR}/PaperDeepSearch--Model-Qwen35-122B-SFT--RefineSummary--PassAt1-MaxTurn400--Range0to2000--0616.jsonl"
  "${SFT_DATA_DIR}/PaperWideSearch--Model-Qwen35-122B-Merge--RefineSummary--PassAt1-Range0to500--0615.jsonl"
  "${KEYTURN_MULTILOSS_ROOT}/easy_direct_multiloss.jsonl"
  "${KEYTURN_MULTILOSS_ROOT}/easy_summary_multiloss.jsonl"
  "${KEYTURN_MULTILOSS_ROOT}/medium_direct_multiloss.jsonl"
  "${KEYTURN_MULTILOSS_ROOT}/medium_summary_multiloss.jsonl"
  "${KEYTURN_MULTILOSS_ROOT}/hard_summary_multiloss.jsonl"
)

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

is_truthy() {
  [[ "${1:-}" == "1" || "${1:-}" == "true" || "${1:-}" == "yes" ]]
}

activate_environment() {
  if [[ -f "${SWIFT_ENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${SWIFT_ENV}/bin/activate"
  elif [[ -f "/root/anaconda3/bin/activate" ]]; then
    # Keep the entry point compatible with the 0804 scripts.
    # shellcheck source=/dev/null
    source "/root/anaconda3/bin/activate" "${SWIFT_ENV}"
  else
    die "cannot activate SWIFT_ENV=${SWIFT_ENV}; run prepare_dpsk_env.sh first"
  fi
}

[[ "${NNODES}" -eq 8 ]] || die "expected NNODES=8, got ${NNODES}"
[[ "${GPUS_PER_NODE}" -eq 8 ]] || die "expected GPUS_PER_NODE=8, got ${GPUS_PER_NODE}"
[[ "${NODE_RANK}" -ge 0 && "${NODE_RANK}" -lt "${NNODES}" ]] || die "invalid NODE_RANK=${NODE_RANK}"
[[ -f "${MODEL_PATH}/config.json" ]] || die "model config not found: ${MODEL_PATH}/config.json"
[[ -d "${MODEL_PATH}" ]] || die "model directory not found: ${MODEL_PATH}"

DATASETS=()
if [[ -n "${DATASET_FILES:-}" ]]; then
  read -r -a DATASETS <<< "${DATASET_FILES}"
else
  DATASETS=("${DEFAULT_DATASETS[@]}")
fi
((${#DATASETS[@]} > 0)) || die "no datasets configured"
for dataset in "${DATASETS[@]}"; do
  [[ -f "${dataset}" ]] || die "dataset not found: ${dataset}"
done

VALIDATION_ARGS=()
if [[ -n "${VAL_DATASET}" ]]; then
  [[ -f "${VAL_DATASET}" ]] || die "validation dataset not found: ${VAL_DATASET}"
  VALIDATION_ARGS=(--val_dataset "${VAL_DATASET}")
fi

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
MODEL_PARALLEL_SIZE=$((TP * PP * CP))
((WORLD_SIZE % MODEL_PARALLEL_SIZE == 0)) \
  || die "WORLD_SIZE=${WORLD_SIZE} is not divisible by TP*PP*CP=${MODEL_PARALLEL_SIZE}"
DENSE_DP=$((WORLD_SIZE / MODEL_PARALLEL_SIZE))
EXPERT_MODEL_PARALLEL_SIZE=$((ETP * EP * PP))
((WORLD_SIZE % EXPERT_MODEL_PARALLEL_SIZE == 0)) \
  || die "WORLD_SIZE=${WORLD_SIZE} is not divisible by ETP*EP*PP=${EXPERT_MODEL_PARALLEL_SIZE}"
EXPERT_DP=$((WORLD_SIZE / EXPERT_MODEL_PARALLEL_SIZE))
BATCH_UNIT=$((MICRO_BATCH_SIZE * DENSE_DP))
((GLOBAL_BATCH_SIZE % BATCH_UNIT == 0)) \
  || die "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} is not divisible by micro_batch*dense_dp=${BATCH_UNIT}"
((EP > 0 && 256 % EP == 0)) || die "EP=${EP} must divide the 256 routed experts in this checkpoint"
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / BATCH_UNIT))

[[ "${TRAIN_TYPE}" == "full" ]] || die "this script only supports TRAIN_TYPE=full"

export NNODES NPROC_PER_NODE="${GPUS_PER_NODE}" NODE_RANK MASTER_ADDR MASTER_PORT
export MODEL_PATH
export SWIFT_USE_MCORE_GDN=1
export SKIP_MULTIMODAL_MTP_VALIDATION=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107,mlx5_108}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MEGATRON_DSV4_UNFUSED_ATTN_CHUNK_ROWS="${DSV4_UNFUSED_ATTN_CHUNK_ROWS}"
if [[ -d "${NCCL_SHARED_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${NCCL_SHARED_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export MODELSCOPE_CACHE="${CACHE_ROOT_BASE}/node${NODE_RANK}"
export HF_HOME="${MODELSCOPE_CACHE}/huggingface"
export HF_DATASETS_CACHE="${MODELSCOPE_CACHE}/datasets"
mkdir -p \
  "${OUTPUT_DIR}" \
  "${TENSORBOARD_DIR}" \
  "${LOG_DIR}" \
  "${MODELSCOPE_CACHE}" \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_DATASETS_CACHE}/map_cache" \
  "${MODELSCOPE_CACHE}/lockers"

activate_environment
command -v megatron >/dev/null 2>&1 || die "megatron command not found in ${SWIFT_ENV}"
PYTHON_BIN="$(command -v python)"

"${PYTHON_BIN}" - "${MODEL_PATH}" <<'PY'
import sys

from transformers import AutoConfig

model_path = sys.argv[1]
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
if config.model_type != 'deepseek_v4':
    raise SystemExit(f'expected model_type=deepseek_v4, got {config.model_type!r}')

import mcore_bridge  # noqa: F401
import swift  # noqa: F401
PY

MTP_ARGS=()
if [[ "${MTP_NUM_LAYERS}" != "0" ]]; then
  MTP_ARGS=(--mtp_num_layers "${MTP_NUM_LAYERS}")
fi

TRAIN_DURATION_ARGS=()
if [[ -n "${TRAIN_ITERS}" ]]; then
  [[ "${TRAIN_ITERS}" -gt 0 ]] || die "TRAIN_ITERS must be positive, got ${TRAIN_ITERS}"
  TRAIN_DURATION_ARGS=(--train_iters "${TRAIN_ITERS}")
else
  TRAIN_DURATION_ARGS=(--num_train_epochs "${NUM_TRAIN_EPOCHS}")
fi

TRAIN_CMD=(
  megatron sft
  --model "${MODEL_PATH}"
  --model_type deepseek_v4
  --template deepseek_v4_flash
  --bridge_backend mcore-bridge
  --save_safetensors true
  --max_shard_size 1GB
  --no_save_optim true
  --no_save_rng true
  --dataset "${DATASETS[@]}"
  "${VALIDATION_ARGS[@]}"
  --load_from_cache_file true
  --add_non_thinking_prefix true
  --loss_scale "${LOSS_SCALE}"
  --split_dataset_ratio 0.0
  --tensor_model_parallel_size "${TP}"
  --pipeline_model_parallel_size "${PP}"
  --pipeline_model_parallel_layout "${PIPELINE_MODEL_PARALLEL_LAYOUT}"
  --expert_model_parallel_size "${EP}"
  --expert_tensor_parallel_size "${ETP}"
  --context_parallel_size "${CP}"
  --sequence_parallel true
  --micro_batch_size "${MICRO_BATCH_SIZE}"
  --global_batch_size "${GLOBAL_BATCH_SIZE}"
  --recompute_granularity full
  --recompute_method uniform
  --recompute_num_layers "${RECOMPUTE_NUM_LAYERS}"
  "${MTP_ARGS[@]}"
  "${TRAIN_DURATION_ARGS[@]}"
  --finetune true
  --cross_entropy_loss_fusion true
  --lr "${LR}"
  --lr_warmup_fraction "${LR_WARMUP_FRACTION}"
  --min_lr "${MIN_LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --output_dir "${OUTPUT_DIR}"
  --tensorboard_dir "${TENSORBOARD_DIR}"
  --eval_steps "${EVAL_STEPS}"
  --eval_iters "${EVAL_ITERS}"
  --save_steps "${SAVE_STEPS}"
  --logging_steps "${LOGGING_STEPS}"
  --max_length "${MAX_LENGTH}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --dataset_num_proc "${DATASET_NUM_PROC}"
  --attention_backend flash
  --attn_impl flash_attn
  --padding_free false
  --group_by_length "${GROUP_BY_LENGTH}"
  --packing "${PACKING}"
  --packing_length "${PACKING_LENGTH}"
  --packing_num_proc "${PACKING_NUM_PROC}"
  --sequence_packing_scheduler "${SEQUENCE_PACKING_SCHEDULER}"
  --cp_partition_mode "${CP_PARTITION_MODE}"
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}"
  --dataloader_pin_memory "${DATALOADER_PIN_MEMORY}"
  --dataloader_persistent_workers "${DATALOADER_PERSISTENT_WORKERS}"
  --manual_gc "${MANUAL_GC}"
  --manual_gc_steps "${MANUAL_GC_STEPS}"
  --moe_permute_fusion true
  --moe_grouped_gemm true
  --moe_shared_expert_overlap true
  --moe_aux_loss_coeff "${MOE_AUX_LOSS_COEFF}"
  --use-distributed-optimizer true
  --fp8_recipe blockwise
  --fp8_format e4m3
  --fp8_param_gather true
  --freeze_llm false
  --tuner_type full
  --optimizer_cpu_offload true
  # Megatron's CPU-offload optimizer currently rejects the BF16 sharded views
  # created by --use_precision_aware_optimizer ("can't optimize a non-leaf
  # Tensor"). Keep it disabled and let HybridDeviceOptimizer maintain its FP32
  # CPU master copies instead.
  --use_precision_aware_optimizer false
  --optimizer_offload_fraction "${OPTIMIZER_OFFLOAD_FRACTION}"
  --main_grads_dtype "${MAIN_GRADS_DTYPE}"
)

LOG_FILE="${LOG_DIR}/node${NODE_RANK}.log"
printf 'DeepSeek-V4-Flash-0731 full SFT node %s/%s\n' "${NODE_RANK}" "${NNODES}"
printf 'MODEL=%s\nOUTPUT=%s\n' "${MODEL_PATH}" "${OUTPUT_DIR}"
printf 'MASTER_ADDR=%s MASTER_PORT=%s WORLD_SIZE=%s\n' "${MASTER_ADDR}" "${MASTER_PORT}" "${WORLD_SIZE}"
printf 'TP=%s PP=%s EP=%s ETP=%s CP=%s dense_DP=%s expert_DP=%s grad_accum=%s MTP=%s\n' \
  "${TP}" "${PP}" "${EP}" "${ETP}" "${CP}" "${DENSE_DP}" "${EXPERT_DP}" "${GRADIENT_ACCUMULATION_STEPS}" "${MTP_NUM_LAYERS}"
printf 'PIPELINE_MODEL_PARALLEL_LAYOUT=%s\n' "${PIPELINE_MODEL_PARALLEL_LAYOUT}"
printf 'MAX_LENGTH=%s PACKING=%s PACKING_LENGTH=%s LR=%s MIN_LR=%s EPOCHS=%s\n' \
  "${MAX_LENGTH}" "${PACKING}" "${PACKING_LENGTH}" "${LR}" "${MIN_LR}" "${NUM_TRAIN_EPOCHS}"
printf 'DATASETS:\n'
printf '  %s\n' "${DATASETS[@]}"
printf 'CMD: %q ' "${TRAIN_CMD[@]}" > "${LOG_FILE}"
printf '\n' >> "${LOG_FILE}"

if is_truthy "${DRY_RUN}"; then
  printf 'DRY_RUN=1; command not executed.\n'
  exit 0
fi

"${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
