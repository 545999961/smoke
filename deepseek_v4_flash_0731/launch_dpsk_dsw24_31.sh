#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V15_ROOT="${V15_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/train_dpsk.sh}"
PREPARE_SCRIPT="${PREPARE_SCRIPT:-${SCRIPT_DIR}/prepare_dpsk_env.sh}"
SSH_CONFIG="${SSH_CONFIG:-/share/project/chaofan/code/self_evolving/connect_dsw/config}"
SESSION_NAME="${SESSION_NAME:-train_0805_dpsk_v4_flash_0731}"
MASTER_PORT="${MASTER_PORT_OVERRIDE:-29845}"
MASTER_ADDR_OVERRIDE="${MASTER_ADDR_OVERRIDE:-}"
SWIFT_ENV="${SWIFT_ENV:-/share/project/chaofan/envs/swift_dsv4_0805_updated}"
MODEL_PATH="${MODEL_PATH:-/share/project/shared_models/DeepSeek-V4-Flash-0731}"
RUN_TAG="${RUN_TAG:-dpsk_v4_flash_0731_full_sft}"
OUTPUT_DIR="${OUTPUT_DIR:-${V15_ROOT}/train/result/0805_dpsk/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/${RUN_TAG}}"
CACHE_ROOT_BASE="${CACHE_ROOT_BASE:-/share/project/chaofan/cache/modelscope/0805_dpsk/${RUN_TAG}}"
VAL_DATASET="${VAL_DATASET-/share/project/chaofan/code/self_evolving/unify_browsecomp/results/collect/ws_processed/0501/validation/valid.jsonl}"
SFT_DATA_DIR="${SFT_DATA_DIR:-/share/project/kunluo/Projects/ScienceAgent/GeneralSearchAgent/SearchAgent-SLM/dataset/SFT}"
CLAUDE_BC_DATA_DIR="${CLAUDE_BC_DATA_DIR:-/share/project/chaofan/code/self_evolving/unify_browsecomp/scripts/0514/keyturn_loss_eval_data}"
KEYTURN_MULTILOSS_ROOT="${KEYTURN_MULTILOSS_ROOT:-${V15_ROOT}/data/source/previous-trajectories/train_data_key_turn_multiloss/previous-sft-122b_bc}"

TP="${TP:-1}"
PP="${PP:-16}"
EP="${EP:-4}"
ETP="${ETP:-1}"
CP="${CP:-4}"
PIPELINE_MODEL_PARALLEL_LAYOUT="${PIPELINE_MODEL_PARALLEL_LAYOUT:-Et*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*2|t*2|t*2|t*2|t*2L}"
# DeepSeek-V4's DSpark speculative-decoding blocks are not Megatron generic
# MTP layers, so they are deliberately excluded from SFT (see new env guard).
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-0}"
MAIN_GRADS_DTYPE="${MAIN_GRADS_DTYPE:-fp32}"
TRAIN_TYPE="${TRAIN_TYPE:-full}"
MAX_LENGTH="${MAX_LENGTH:-65536}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
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

DEFAULT_HOSTS=(dsw-24 dsw-25 dsw-26 dsw-27 dsw-28 dsw-29 dsw-30 dsw-31)
HOSTS_LIST=()
SKIP_BUSY_CHECK="${SKIP_BUSY_CHECK:-false}"
DRY_RUN=false
PREPARE_ENV=false
PREFLIGHT_ONLY=false

DATASET_FILES="${DATASET_FILES:-${SFT_DATA_DIR}/browsecomp_full.jsonl ${CLAUDE_BC_DATA_DIR}/claude_browsecomp_kstep_non_react_loss_eval.jsonl ${CLAUDE_BC_DATA_DIR}/claude_browsecomp_summary_all_non_react_loss_eval.jsonl ${CLAUDE_BC_DATA_DIR}/claude_browsecomp_kstep_react_loss_eval.jsonl ${CLAUDE_BC_DATA_DIR}/claude_browsecomp_summary_all_react_loss_eval.jsonl ${SFT_DATA_DIR}/processed_64k_hle_w_tool_filter_train.jsonl ${SFT_DATA_DIR}/PaperDeepSearch--Model-Qwen35-122B-SFT--Direct--PassAt1-Range0to2000--0614--DefaultSystemPrompt.jsonl ${SFT_DATA_DIR}/PaperDeepSearch--Model-Qwen35-122B-SFT--RefineSummary--PassAt1-MaxTurn400--Range0to2000--0616.jsonl ${SFT_DATA_DIR}/PaperWideSearch--Model-Qwen35-122B-Merge--RefineSummary--PassAt1-Range0to500--0615.jsonl ${KEYTURN_MULTILOSS_ROOT}/easy_direct_multiloss.jsonl ${KEYTURN_MULTILOSS_ROOT}/easy_summary_multiloss.jsonl ${KEYTURN_MULTILOSS_ROOT}/medium_direct_multiloss.jsonl ${KEYTURN_MULTILOSS_ROOT}/medium_summary_multiloss.jsonl ${KEYTURN_MULTILOSS_ROOT}/hard_summary_multiloss.jsonl}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

is_truthy() {
  [[ "${1:-}" == "1" || "${1:-}" == "true" || "${1:-}" == "yes" ]]
}

ssh_cfg() {
  ssh -n -F "${SSH_CONFIG}" \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new \
    -o UpdateHostKeys=no \
    "$@"
}

usage() {
  cat <<EOF
Usage: bash $(basename "$0") [--prepare-env] [--preflight-only] [--dry-run]

Default hosts: ${DEFAULT_HOSTS[*]}
Default model: ${MODEL_PATH}
Default Swift environment: ${SWIFT_ENV}

  --prepare-env       prepare the dedicated DeepSeek-V4 environment once locally
  --preflight-only    SSH-check all eight hosts but do not create tmux sessions
  --dry-run           validate local paths and print the selected hosts; do not SSH
  --skip-busy-check   allow launch when a training process is detected

Overrides:
  HOSTS="dsw-24 dsw-25 ..."  replace the eight default hosts
  MASTER_PORT_OVERRIDE=...  change the rendezvous port
  MAX_LENGTH=65536          raise sequence length after a short smoke run
  DATASET_FILES="..."       replace the default space-separated JSONL list
EOF
}

parse_args() {
  while (($# > 0)); do
    case "$1" in
      --prepare-env)
        PREPARE_ENV=true
        ;;
      --dry-run)
        DRY_RUN=true
        ;;
      --preflight-only)
        PREFLIGHT_ONLY=true
        ;;
      --skip-busy-check)
        SKIP_BUSY_CHECK=true
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown option: $1"
        ;;
    esac
    shift
  done
}

build_hosts() {
  if [[ -n "${HOSTS:-}" ]]; then
    read -r -a HOSTS_LIST <<< "${HOSTS}"
  else
    HOSTS_LIST=("${DEFAULT_HOSTS[@]}")
  fi
  ((${#HOSTS_LIST[@]} == 8)) || die "expected exactly 8 hosts, got ${#HOSTS_LIST[@]}"
}

host_is_configured() {
  ssh -G -F "${SSH_CONFIG}" "$1" >/dev/null 2>&1
}

resolve_target_key() {
  ssh -G -F "${SSH_CONFIG}" "$1" | awk '
    $1 == "hostname" { hostname = $2 }
    $1 == "user" { user = $2 }
    $1 == "port" { port = $2 }
    END {
      if (hostname == "" || user == "" || port == "") exit 1
      printf "%s@%s:%s\n", user, hostname, port
    }
  '
}

validate_unique_targets() {
  local host target
  declare -A seen=()

  for host in "${HOSTS_LIST[@]}"; do
    host_is_configured "${host}" || die "host is not configured in ${SSH_CONFIG}: ${host}"
    target="$(resolve_target_key "${host}")"
    [[ -z "${seen[${target}]:-}" ]] \
      || die "${host} and ${seen[${target}]} resolve to the same SSH target: ${target}"
    seen["${target}"]="${host}"
  done
}

validate_local_paths() {
  local dataset
  [[ -f "${SSH_CONFIG}" ]] || die "SSH config not found: ${SSH_CONFIG}"
  [[ -f "${TRAIN_SCRIPT}" ]] || die "train script not found: ${TRAIN_SCRIPT}"
  [[ -f "${MODEL_PATH}/config.json" ]] || die "model config not found: ${MODEL_PATH}/config.json"
  [[ -d "${MODEL_PATH}" ]] || die "model directory not found: ${MODEL_PATH}"
  if [[ -n "${VAL_DATASET}" ]]; then
    [[ -f "${VAL_DATASET}" ]] || die "validation dataset not found: ${VAL_DATASET}"
  fi
  read -r -a datasets <<< "${DATASET_FILES}"
  ((${#datasets[@]} > 0)) || die "no datasets configured"
  for dataset in "${datasets[@]}"; do
    [[ -f "${dataset}" ]] || die "dataset not found: ${dataset}"
  done
}

remote_has_training() {
  ssh_cfg "$1" \
    "ps -eo args= | awk '
      /swift\\/cli\\/_meg[a]tron\\/sft[.]py|to[r]ch[.]distributed[.]run|\\/meg[a]tron sft/ {
        if (\$0 !~ /awk/) found=1
      }
      END { exit(found ? 0 : 1) }
    '" >/dev/null
}

remote_dsv4_check() {
  local host="$1"
  ssh_cfg "${host}" \
    "set -e
     test -x '${SWIFT_ENV}/bin/python'
     test -x '${SWIFT_ENV}/bin/megatron'
     test -f '${MODEL_PATH}/config.json'
     '${SWIFT_ENV}/bin/python' -c \"from transformers import AutoConfig; c=AutoConfig.from_pretrained('${MODEL_PATH}', trust_remote_code=True); assert c.model_type == 'deepseek_v4', c.model_type; import mcore_bridge, swift\"" \
    >/dev/null
}

remote_preflight() {
  local host="$1"
  ssh_cfg "${host}" "command -v tmux >/dev/null" \
    || die "tmux is unavailable on ${host}"
  if ssh_cfg "${host}" "tmux has-session -t '${SESSION_NAME}' 2>/dev/null"; then
    die "tmux session already exists on ${host}: ${SESSION_NAME}"
  fi
  if ! is_truthy "${SKIP_BUSY_CHECK}" && remote_has_training "${host}"; then
    die "${host} already has a Megatron/torch distributed training process; nothing was launched"
  fi
  remote_dsv4_check "${host}" \
    || die "DeepSeek-V4 environment preflight failed on ${host}; run --prepare-env on the shared filesystem"
}

resolve_master_addr() {
  if [[ -n "${MASTER_ADDR_OVERRIDE}" ]]; then
    printf '%s\n' "${MASTER_ADDR_OVERRIDE}"
  else
    ssh_cfg "${HOSTS_LIST[0]}" "hostname -I | awk '{print \$1}'"
  fi
}

launch_one() {
  local host="$1"
  local node_rank="$2"
  local master_addr="$3"
  local export_line

  export_line="export MASTER_ADDR='${master_addr}' MASTER_PORT='${MASTER_PORT}' NNODES='8' NODE_RANK='${node_rank}' V15_ROOT='${V15_ROOT}' SWIFT_ENV='${SWIFT_ENV}' MODEL_PATH='${MODEL_PATH}' RUN_TAG='${RUN_TAG}' OUTPUT_DIR='${OUTPUT_DIR}' LOG_DIR='${LOG_DIR}' CACHE_ROOT_BASE='${CACHE_ROOT_BASE}' VAL_DATASET='${VAL_DATASET}' DATASET_FILES='${DATASET_FILES}' TP='${TP}' PP='${PP}' EP='${EP}' ETP='${ETP}' CP='${CP}' PIPELINE_MODEL_PARALLEL_LAYOUT='${PIPELINE_MODEL_PARALLEL_LAYOUT}' MTP_NUM_LAYERS='${MTP_NUM_LAYERS}' MAIN_GRADS_DTYPE='${MAIN_GRADS_DTYPE}' TRAIN_TYPE='${TRAIN_TYPE}' MAX_LENGTH='${MAX_LENGTH}' NUM_TRAIN_EPOCHS='${NUM_TRAIN_EPOCHS}' TRAIN_ITERS='${TRAIN_ITERS}' MICRO_BATCH_SIZE='${MICRO_BATCH_SIZE}' GLOBAL_BATCH_SIZE='${GLOBAL_BATCH_SIZE}' LR='${LR}' MIN_LR='${MIN_LR}' LR_WARMUP_FRACTION='${LR_WARMUP_FRACTION}' WEIGHT_DECAY='${WEIGHT_DECAY}' LOSS_SCALE='${LOSS_SCALE}' OPTIMIZER_OFFLOAD_FRACTION='${OPTIMIZER_OFFLOAD_FRACTION}' RECOMPUTE_NUM_LAYERS='${RECOMPUTE_NUM_LAYERS}' DSV4_UNFUSED_ATTN_CHUNK_ROWS='${DSV4_UNFUSED_ATTN_CHUNK_ROWS}' MOE_AUX_LOSS_COEFF='${MOE_AUX_LOSS_COEFF}' DATALOADER_NUM_WORKERS='${DATALOADER_NUM_WORKERS}' DATASET_NUM_PROC='${DATASET_NUM_PROC}' DATALOADER_PREFETCH_FACTOR='${DATALOADER_PREFETCH_FACTOR}' DATALOADER_PIN_MEMORY='${DATALOADER_PIN_MEMORY}' DATALOADER_PERSISTENT_WORKERS='${DATALOADER_PERSISTENT_WORKERS}' PACKING='${PACKING}' PACKING_LENGTH='${PACKING_LENGTH}' PACKING_NUM_PROC='${PACKING_NUM_PROC}' GROUP_BY_LENGTH='${GROUP_BY_LENGTH}' SEQUENCE_PACKING_SCHEDULER='${SEQUENCE_PACKING_SCHEDULER}' CP_PARTITION_MODE='${CP_PARTITION_MODE}' MANUAL_GC='${MANUAL_GC}' MANUAL_GC_STEPS='${MANUAL_GC_STEPS}' SAVE_STEPS='${SAVE_STEPS}' EVAL_STEPS='${EVAL_STEPS}' EVAL_ITERS='${EVAL_ITERS}' LOGGING_STEPS='${LOGGING_STEPS}'"

  log "launching rank ${node_rank} on ${host}"
  ssh_cfg "${host}" \
    "set -euo pipefail
     mkdir -p '${LOG_DIR}'
     tmux new-session -d -s '${SESSION_NAME}' -c '${SCRIPT_DIR}'
     tmux set-option -t '${SESSION_NAME}' remain-on-exit off >/dev/null 2>&1 || true
     tmux send-keys -t '${SESSION_NAME}' \"cd '${SCRIPT_DIR}'\" C-m
     tmux send-keys -t '${SESSION_NAME}' \"unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK DRY_RUN; ${export_line}\" C-m
     tmux send-keys -t '${SESSION_NAME}' \"bash '${TRAIN_SCRIPT}'\" C-m
    "
}

main() {
  local host master_addr rank

  parse_args "$@"
  build_hosts
  validate_local_paths

  log "hosts=${HOSTS_LIST[*]}"
  log "model=${MODEL_PATH}"
  log "output=${OUTPUT_DIR}"
  log "topology=8 nodes x 8 GPUs, TP=${TP} PP=${PP} EP=${EP} ETP=${ETP} CP=${CP} MTP=${MTP_NUM_LAYERS} MAX_LENGTH=${MAX_LENGTH}"

  if is_truthy "${PREPARE_ENV}"; then
    [[ -f "${PREPARE_SCRIPT}" ]] || die "environment preparation script not found: ${PREPARE_SCRIPT}"
    log "preparing shared environment ${SWIFT_ENV}"
    SWIFT_ENV="${SWIFT_ENV}" MODEL_PATH="${MODEL_PATH}" bash "${PREPARE_SCRIPT}"
  fi

  if is_truthy "${DRY_RUN}"; then
    log "dry-run only: no SSH connection was made and nothing was launched"
    return
  fi

  command -v ssh >/dev/null || die "ssh is unavailable"
  validate_unique_targets
  for host in "${HOSTS_LIST[@]}"; do
    remote_preflight "${host}"
  done
  if is_truthy "${PREFLIGHT_ONLY}"; then
    log "remote preflight passed on all eight hosts; nothing was launched"
    return
  fi

  master_addr="$(resolve_master_addr)"
  [[ -n "${master_addr}" ]] || die "failed to resolve MASTER_ADDR from ${HOSTS_LIST[0]}"
  log "MASTER_ADDR=${master_addr} MASTER_PORT=${MASTER_PORT} NNODES=8"

  for rank in "${!HOSTS_LIST[@]}"; do
    launch_one "${HOSTS_LIST[${rank}]}" "${rank}" "${master_addr}"
  done
  log "all ranks submitted; no existing process was killed"
}

main "$@"
