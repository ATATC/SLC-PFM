#!/bin/bash
#SBATCH --job-name=slc-pfm-distill
#SBATCH --account=rrg-jma
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=124G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"
echo "---"

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR}}"
INPUT_ROOT="${INPUT_ROOT:-/project/rrg-jma/shared/SLC-PFM}"
FEATURE_ROOT="${FEATURE_ROOT:-/project/rrg-jma/shared/SLC-PFM_features}"
TOKEN_FEATURE_ROOT="${TOKEN_FEATURE_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/project/rrg-jma/shared/SLC-PFM_distill/cradio_v4_so400m_virchow_hoptimus_uni}"
HF_HOME="${HF_HOME:-/project/rrg-jma/${USER}/hf_cache}"
TORCH_HOME="${TORCH_HOME:-/project/rrg-jma/${USER}/torch_cache}"
ENCODERS="${ENCODERS:-virchow2,hoptimus1,uni_v2}"
RADIO_VERSION="${RADIO_VERSION:-c-radio_v4-so400m}"
RADIO_REPO="${RADIO_REPO:-NVlabs/RADIO}"
RADIO_SOURCE="${RADIO_SOURCE:-github}"
RADIO_FORCE_RELOAD="${RADIO_FORCE_RELOAD:-0}"
VITDET_WINDOW_SIZE="${VITDET_WINDOW_SIZE:-}"
ONLINE_TOKEN_TEACHERS="${ONLINE_TOKEN_TEACHERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-}"
if [[ -z "${BATCH_SIZE}" ]]; then
  if [[ "${ONLINE_TOKEN_TEACHERS}" != "0" ]]; then
    BATCH_SIZE=4
  else
    BATCH_SIZE=64
  fi
fi
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-100000}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
TILE_SIZE="${TILE_SIZE:-224}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-20}"
LIMIT_ZIPS="${LIMIT_ZIPS:-}"
STATS_MAX_FILES="${STATS_MAX_FILES:-}"
INCLUDE_TOKEN_MAPS="${INCLUDE_TOKEN_MAPS:-0}"
SPATIAL_LOSS_WEIGHT="${SPATIAL_LOSS_WEIGHT:-1.0}"
NVIDIA_SMI_GPU_MONITOR="${NVIDIA_SMI_GPU_MONITOR:-1}"
NVIDIA_SMI_GPU_MONITOR_SAMPLE_SECONDS="${NVIDIA_SMI_GPU_MONITOR_SAMPLE_SECONDS:-5}"
NVIDIA_SMI_GPU_MONITOR_REPORT_SECONDS="${NVIDIA_SMI_GPU_MONITOR_REPORT_SECONDS:-60}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"

export HF_HOME
export TORCH_HOME
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "${CODE_DIR}"
mkdir -p "${OUTPUT_DIR}" "${HF_HOME}" "${TORCH_HOME}"

module load python/3.12
module load arrow
module load cuda/13.2
module load opencv
module load openslide
source /scratch/atatc/venv_slcpfm/bin/activate

monitor_pid=""
cleanup() {
  if [[ -n "${monitor_pid}" ]] && kill -0 "${monitor_pid}" 2>/dev/null; then
    echo "Stopping nvidia-smi GPU monitor pid=${monitor_pid}"
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT TERM INT

if [[ "${NVIDIA_SMI_GPU_MONITOR}" != "0" ]]; then
  monitor_label="${SLURM_JOB_ID}_distill"
  echo "Starting nvidia-smi GPU monitor: label=${monitor_label}"
  python -u scripts/nvidia_smi_gpu_monitor.py \
    --label "${monitor_label}" \
    --sample-interval-seconds "${NVIDIA_SMI_GPU_MONITOR_SAMPLE_SECONDS}" \
    --report-interval-seconds "${NVIDIA_SMI_GPU_MONITOR_REPORT_SECONDS}" \
    --nvidia-smi-bin "${NVIDIA_SMI_BIN}" &
  monitor_pid="$!"
fi

cmd=(
  python -u scripts/train_cradio_distill.py
  --input-root "${INPUT_ROOT}"
  --feature-root "${FEATURE_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --encoders "${ENCODERS}"
  --radio-version "${RADIO_VERSION}"
  --radio-repo "${RADIO_REPO}"
  --radio-source "${RADIO_SOURCE}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --max-steps "${MAX_STEPS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --tile-size "${TILE_SIZE}"
  --amp-dtype "${AMP_DTYPE}"
  --save-every "${SAVE_EVERY}"
  --log-every "${LOG_EVERY}"
  --spatial-loss-weight "${SPATIAL_LOSS_WEIGHT}"
)

if [[ -n "${TOKEN_FEATURE_ROOT}" ]]; then
  cmd+=(--token-feature-root "${TOKEN_FEATURE_ROOT}")
fi

if [[ "${RADIO_FORCE_RELOAD}" != "0" ]]; then
  cmd+=(--radio-force-reload)
fi

if [[ -n "${VITDET_WINDOW_SIZE}" ]]; then
  cmd+=(--vitdet-window-size "${VITDET_WINDOW_SIZE}")
fi

if [[ -n "${LIMIT_ZIPS}" ]]; then
  cmd+=(--limit-zips "${LIMIT_ZIPS}")
fi

if [[ -n "${STATS_MAX_FILES}" ]]; then
  cmd+=(--stats-max-files "${STATS_MAX_FILES}")
fi

if [[ "${INCLUDE_TOKEN_MAPS}" != "0" ]]; then
  cmd+=(--include-token-maps)
fi

if [[ "${ONLINE_TOKEN_TEACHERS}" != "0" ]]; then
  cmd+=(--online-token-teachers)
fi

echo "Command:"
printf '  %q' "${cmd[@]}"
echo

"${cmd[@]}"

echo "---"
echo "End: $(date)"
