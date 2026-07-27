#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_PROFILE="${1:-${CONTEXT_PROFILE:-64k}}"

if [[ "$#" -gt 1 ]]; then
  printf 'usage: %s [64k|200k]\n' "$0" >&2
  exit 2
fi

case "${CONTEXT_PROFILE}" in
  64k)
    DEFAULT_TRAIN_DATASET="${SCRIPT_DIR}/data/long_64k_train.jsonl"
    DEFAULT_MAX_LENGTH=65536
    DEFAULT_CP=1
    DEFAULT_GLOBAL_BATCH_SIZE=128
    DEFAULT_NUM_TRAIN_EPOCHS=2
    ;;
  200k)
    DEFAULT_TRAIN_DATASET="${SCRIPT_DIR}/data/long_256k_train.jsonl"
    DEFAULT_MAX_LENGTH=262144
    DEFAULT_CP=2
    DEFAULT_GLOBAL_BATCH_SIZE=1
    DEFAULT_NUM_TRAIN_EPOCHS=1
    ;;
  *)
    printf 'usage: %s [64k|200k]\n' "$0" >&2
    exit 2
    ;;
esac

activate_requested_env() {
  if [[ -z "${SWIFT_ENV:-}" ]]; then
    return
  fi
  if [[ -f "${SWIFT_ENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${SWIFT_ENV}/bin/activate"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck source=/dev/null
    source "${conda_base}/etc/profile.d/conda.sh"
    conda activate "${SWIFT_ENV}"
    return
  fi
  printf 'ERROR: cannot activate SWIFT_ENV=%s\n' "${SWIFT_ENV}" >&2
  exit 1
}

activate_requested_env

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

is_truthy() {
  [[ "${1:-}" == "1" || "${1:-}" == "true" || "${1:-}" == "yes" ]]
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

read_datasets() {
  local line
  DATASETS=()
  if [[ -n "${DATASET_MANIFEST:-}" ]]; then
    [[ -f "${DATASET_MANIFEST}" ]] || die "manifest not found: ${DATASET_MANIFEST}"
    while IFS= read -r line || [[ -n "${line}" ]]; do
      line="$(trim "${line}")"
      [[ -z "${line}" || "${line:0:1}" == "#" ]] && continue
      DATASETS+=("${line}")
    done < "${DATASET_MANIFEST}"
  else
    DATASETS+=("${TRAIN_DATASET}")
  fi
  [[ "${#DATASETS[@]}" -gt 0 ]] || die "no training datasets configured"
}

read_model_metadata() {
  python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    config = json.load(stream)
text = config.get("text_config", {})
keys = ("num_hidden_layers", "num_attention_heads", "num_key_value_heads", "num_experts")
print(" ".join(str(text.get(key, config.get(key, 0)) or 0) for key in keys))
PY
}

validate_parallelism() {
  local world_size model_parallel dense_dp batch_unit
  (( NNODES > 0 )) || die "NNODES must be positive"
  (( NPROC_PER_NODE > 0 )) || die "NPROC_PER_NODE must be positive"
  (( NODE_RANK >= 0 && NODE_RANK < NNODES )) \
    || die "NODE_RANK=${NODE_RANK} must be in [0, $((NNODES - 1))]"
  (( TP > 0 && PP > 0 && EP > 0 && ETP > 0 && CP > 0 )) \
    || die "all parallel sizes must be positive"
  world_size=$((NNODES * NPROC_PER_NODE))
  model_parallel=$((TP * PP * CP))
  (( world_size % model_parallel == 0 )) \
    || die "WORLD_SIZE=${world_size} is not divisible by TP*PP*CP=${model_parallel}"
  dense_dp=$((world_size / model_parallel))
  batch_unit=$((MICRO_BATCH_SIZE * dense_dp))
  (( GLOBAL_BATCH_SIZE % batch_unit == 0 )) \
    || die "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} is not divisible by micro_batch*dense_dp=${batch_unit}"
  (( TP % ETP == 0 )) || die "TP=${TP} is not divisible by ETP=${ETP}"

  if [[ -f "${MODEL_PATH}/config.json" ]]; then
    local layers heads kv_heads experts
    read -r layers heads kv_heads experts < <(
      read_model_metadata "${MODEL_PATH}/config.json"
    )
    (( layers > 0 && layers % PP == 0 )) \
      || die "num_hidden_layers=${layers} is not divisible by PP=${PP}"
    (( heads > 0 && heads % TP == 0 )) \
      || die "num_attention_heads=${heads} is not divisible by TP=${TP}"
    (( kv_heads > 0 && (kv_heads % TP == 0 || TP % kv_heads == 0) )) \
      || die "num_key_value_heads=${kv_heads} is incompatible with TP=${TP}"
    (( experts > 0 && experts % EP == 0 )) \
      || die "num_experts=${experts} is not divisible by EP=${EP}"
    if [[ -z "${RECOMPUTE_NUM_LAYERS}" ]]; then
      RECOMPUTE_NUM_LAYERS=$((layers / PP))
    fi
  elif [[ -z "${RECOMPUTE_NUM_LAYERS}" ]]; then
    RECOMPUTE_NUM_LAYERS=12
  fi

  DENSE_DP="${dense_dp}"
  GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / batch_unit))
}

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-122B-A10B}"
TRAIN_DATASET="${TRAIN_DATASET:-${DEFAULT_TRAIN_DATASET}}"
VAL_DATASET="${VAL_DATASET-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/${CONTEXT_PROFILE}}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/reproduction}"
CACHE_ROOT="${CACHE_ROOT:-${SCRIPT_DIR}/.cache/reproduction}"

# Exact defaults from the source 0619 experiment.
NNODES="${NNODES:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
TP="${TP:-8}"
PP="${PP:-4}"
EP="${EP:-8}"
ETP="${ETP:-1}"
CP="${CP:-${DEFAULT_CP}}"
MAX_LENGTH="${MAX_LENGTH:-${DEFAULT_MAX_LENGTH}}"
PACKING_LENGTH="${PACKING_LENGTH:-${MAX_LENGTH}}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${DEFAULT_GLOBAL_BATCH_SIZE}}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-${DEFAULT_NUM_TRAIN_EPOCHS}}"
LR="${LR:-5e-7}"
MIN_LR="${MIN_LR:-1e-7}"
RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-}"

export NNODES NPROC_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT
export SWIFT_USE_MCORE_GDN="${SWIFT_USE_MCORE_GDN:-1}"
export SKIP_MULTIMODAL_MTP_VALIDATION="${SKIP_MULTIMODAL_MTP_VALIDATION:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${CACHE_ROOT}/node${NODE_RANK}/modelscope}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/node${NODE_RANK}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/node${NODE_RANK}/datasets}"
if [[ -n "${NCCL_IB_HCA:-}" ]]; then
  export NCCL_IB_HCA
fi

read_datasets
validate_parallelism

if ! is_truthy "${DRY_RUN:-0}"; then
  if (( NNODES > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
    die "MASTER_ADDR must be a rank-0 address reachable from every node"
  fi
  for dataset in "${DATASETS[@]}"; do
    [[ -f "${dataset}" ]] || die "dataset not found: ${dataset}"
  done
  if [[ -n "${VAL_DATASET}" ]]; then
    [[ -f "${VAL_DATASET}" ]] || die "validation dataset not found: ${VAL_DATASET}"
  fi
fi

TUNER_ARGS=()
if [[ "${TRAIN_TYPE:-full}" == "full" ]]; then
  TUNER_ARGS=(
    --freeze_llm false
    --freeze_vit true
    --freeze_aligner true
    --tuner_type full
    --optimizer_cpu_offload true
    --use_precision_aware_optimizer true
    --optimizer_offload_fraction "${OPTIMIZER_OFFLOAD_FRACTION:-0.80}"
  )
else
  TUNER_ARGS=(
    --tuner_type lora
    --merge_lora true
    --lora_rank "${LORA_RANK:-32}"
    --lora_alpha "${LORA_ALPHA:-64}"
    --target_modules all-linear
  )
fi

TRAIN_CMD=(
  megatron sft
  --model "${MODEL_PATH}"
  --save_safetensors true
  --no_save_optim true
  --no_save_rng true
  --dataset "${DATASETS[@]}"
  --load_from_cache_file true
  --add_non_thinking_prefix true
  --loss_scale "${LOSS_SCALE:-default+ignore_empty_think}"
  --split_dataset_ratio 0.0
  --tensor_model_parallel_size "${TP}"
  --pipeline_model_parallel_size "${PP}"
  --expert_model_parallel_size "${EP}"
  --expert_tensor_parallel_size "${ETP}"
  --context_parallel_size "${CP}"
  --sequence_parallel true
  --micro_batch_size "${MICRO_BATCH_SIZE}"
  --global_batch_size "${GLOBAL_BATCH_SIZE}"
  --recompute_granularity "${RECOMPUTE_GRANULARITY}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --finetune true
  --cross_entropy_loss_fusion true
  --lr "${LR}"
  --lr_warmup_fraction 0.05
  --min_lr "${MIN_LR}"
  --output_dir "${OUTPUT_DIR}"
  --tensorboard_dir "${OUTPUT_DIR}/runs"
  --eval_steps "${EVAL_STEPS:-100}"
  --save_steps "${SAVE_STEPS:-150}"
  --logging_steps "${LOGGING_STEPS:-1}"
  --max_length "${MAX_LENGTH}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-16}"
  --dataset_num_proc "${DATASET_NUM_PROC:-64}"
  --attention_backend "${ATTENTION_BACKEND:-flash}"
  --attn_impl "${ATTN_IMPL:-flash_attn}"
  --padding_free "${PADDING_FREE:-false}"
  --group_by_length "${GROUP_BY_LENGTH:-false}"
  --packing "${PACKING:-true}"
  --packing_length "${PACKING_LENGTH}"
  --moe_grouped_gemm true
  --moe_permute_fusion true
  --moe_shared_expert_overlap true
  --moe_router_dtype fp32
  --moe_aux_loss_coeff 2e-5
  --use-distributed-optimizer true
  --moe_expert_capacity_factor 2
  --enable_channel_loss true
)

if [[ -n "${VAL_DATASET}" ]]; then
  TRAIN_CMD+=(--val_dataset "${VAL_DATASET}")
fi
if [[ "${RECOMPUTE_GRANULARITY}" == "selective" ]]; then
  read -r -a recompute_modules <<< "${RECOMPUTE_MODULES:-core_attn moe}"
  TRAIN_CMD+=(--recompute_modules "${recompute_modules[@]}")
else
  TRAIN_CMD+=(
    --recompute_method "${RECOMPUTE_METHOD}"
    --recompute_num_layers "${RECOMPUTE_NUM_LAYERS}"
  )
fi
TRAIN_CMD+=("${TUNER_ARGS[@]}")

printf 'Reproduction configuration:\n'
printf '  profile=%s model=%s\n  world_size=%s dense_dp=%s grad_accumulation=%s\n' \
  "${CONTEXT_PROFILE}" "${MODEL_PATH}" "$((NNODES * NPROC_PER_NODE))" "${DENSE_DP}" \
  "${GRADIENT_ACCUMULATION_STEPS}"
printf '  TP=%s PP=%s EP=%s ETP=%s CP=%s\n' "${TP}" "${PP}" "${EP}" "${ETP}" "${CP}"
printf '  datasets=%s max_length=%s global_batch=%s epochs=%s\n' \
  "${#DATASETS[@]}" "${MAX_LENGTH}" "${GLOBAL_BATCH_SIZE}" "${NUM_TRAIN_EPOCHS}"
printf 'Command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'

if is_truthy "${DRY_RUN:-0}"; then
  printf 'DRY_RUN enabled; training was not started.\n'
  exit 0
fi

command -v megatron >/dev/null 2>&1 \
  || die "megatron command not found; install ms-swift and Megatron dependencies first"

mkdir -p "${OUTPUT_DIR}/runs" "${LOG_DIR}" \
  "${MODELSCOPE_CACHE}" "${HF_HOME}" "${HF_DATASETS_CACHE}"

"${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_DIR}/node${NODE_RANK}.log"
