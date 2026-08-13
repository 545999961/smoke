#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/share/project/shared_models/DeepSeek-V4-Flash-0731}"
SWIFT_ENV="${SWIFT_ENV:-/share/project/chaofan/envs/swift_dsv4_0805_updated}"
BASE_PYTHON="${BASE_PYTHON:-/share/project/chaofan/envs/swift_0426/bin/python}"
MEGATRON_REF="${MEGATRON_REF:-fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1}"
MCORE_BRIDGE_REF="${MCORE_BRIDGE_REF:-60a0d696d4e95b1c2f3dd560b3109e04dc799c4a}"
MS_SWIFT_REF="${MS_SWIFT_REF:-9298fb8a970aa0d07e362de02aace171cc5acdf5}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.14.1}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/share/project/chaofan/cache/pip/0805_dpsk}"
PATCH_DIR="${PATCH_DIR:-${SCRIPT_DIR}/patches}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ -f "${MODEL_PATH}/config.json" ]] || die "model config not found: ${MODEL_PATH}/config.json"
[[ -x "${BASE_PYTHON}" ]] || die "base Python not found: ${BASE_PYTHON}"
command -v flock >/dev/null 2>&1 || die "flock is required to serialize environment preparation"
command -v patch >/dev/null 2>&1 || die "patch is required to install the runtime fixes"

mkdir -p "$(dirname "${SWIFT_ENV}")" "${PIP_CACHE_DIR}"
exec 9>"${SWIFT_ENV}.lock"
flock 9

if [[ ! -f "${SWIFT_ENV}/pyvenv.cfg" ]]; then
  if [[ -e "${SWIFT_ENV}" ]]; then
    die "${SWIFT_ENV} exists but is not a venv created by this script; choose another SWIFT_ENV"
  fi
  "${BASE_PYTHON}" -m venv --system-site-packages "${SWIFT_ENV}"
fi

PYTHON_BIN="${SWIFT_ENV}/bin/python"
PIP_BIN=("${PYTHON_BIN}" -m pip)
[[ -x "${PYTHON_BIN}" ]] || die "venv Python was not created: ${PYTHON_BIN}"

export PIP_CACHE_DIR
"${PIP_BIN[@]}" install --upgrade pip

# Install the exact revisions from the verified DeepSeek-V4 runtime.
"${PIP_BIN[@]}" install --upgrade \
  "git+https://github.com/NVIDIA/Megatron-LM.git@${MEGATRON_REF}"
"${PIP_BIN[@]}" install --upgrade \
  "git+https://github.com/modelscope/mcore-bridge.git@${MCORE_BRIDGE_REF}"
"${PIP_BIN[@]}" install --upgrade \
  "git+https://github.com/modelscope/ms-swift.git@${MS_SWIFT_REF}"

# The existing shared environment has Transformers 5.2.0, which does not
# register deepseek_v4. Keep this upgrade in the dedicated venv only.
"${PIP_BIN[@]}" install --upgrade "transformers==${TRANSFORMERS_VERSION}"

SITE_PACKAGES="$("${PYTHON_BIN}" - <<'PY'
import sysconfig

print(sysconfig.get_paths()['purelib'])
PY
)"

apply_runtime_patch() {
  local patch_file="${PATCH_DIR}/$1"
  [[ -f "${patch_file}" ]] || die "runtime patch not found: ${patch_file}"
  if patch --batch --forward --dry-run -p1 -d "${SITE_PACKAGES}" < "${patch_file}" >/dev/null; then
    patch --batch --forward -p1 -d "${SITE_PACKAGES}" < "${patch_file}"
  elif patch --batch --reverse --dry-run -p1 -d "${SITE_PACKAGES}" < "${patch_file}" >/dev/null; then
    printf 'Runtime patch already applied: %s\n' "$1"
  else
    die "runtime patch does not match the pinned package: ${patch_file}"
  fi
}

apply_runtime_patch megatron-core.patch
apply_runtime_patch mcore-bridge.patch
apply_runtime_patch ms-swift.patch

"${PYTHON_BIN}" - "${MODEL_PATH}" <<'PY'
import sys

from transformers import AutoConfig

model_path = sys.argv[1]
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
if config.model_type != 'deepseek_v4':
    raise SystemExit(f'expected deepseek_v4, got {config.model_type!r}')

import mcore_bridge  # noqa: F401
import swift  # noqa: F401

print('DeepSeek-V4 environment check passed')
print(f'model_type={config.model_type}')
PY

[[ -x "${SWIFT_ENV}/bin/megatron" ]] \
  || die "megatron entry point was not installed in ${SWIFT_ENV}"

printf 'Prepared DeepSeek-V4 environment: %s\n' "${SWIFT_ENV}"
