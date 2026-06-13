#!/bin/bash
#SBATCH --job-name=slc-pfm-features
#SBATCH --account=rrg-jma
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Node: $(hostname)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"
echo "---"

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR}}"
INPUT_ROOT="${INPUT_ROOT:-/project/rrg-jma/shared/SLC-PFM}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/rrg-jma/shared/SLC-PFM_features}"
HF_HOME="${HF_HOME:-/project/rrg-jma/${USER}/hf_cache}"
ENCODERS="${ENCODERS:-virchow2,hoptimus1,uni_v2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LIMIT_ZIPS="${LIMIT_ZIPS:-}"
LIMIT_TILES="${LIMIT_TILES:-}"
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-4}"

export HF_HOME
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "${CODE_DIR}"
mkdir -p "${OUTPUT_ROOT}" "${HF_HOME}"

module load python/3.12
module load arrow
module load cuda/13.2
module load opencv
module load openslide
source /scratch/atatc/venv_automil/bin/activate

mapfile -t CHUNKS < <(find "${INPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'chunk_*' -printf '%f\n' | sort -V)

if [[ "${#CHUNKS[@]}" -eq 0 ]]; then
  echo "No chunk_* directories found under ${INPUT_ROOT}" >&2
  exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${TASK_ID}" -ge "${#CHUNKS[@]}" ]]; then
  echo "Array task ${TASK_ID} has no matching chunk; found ${#CHUNKS[@]} chunks."
  exit 0
fi

CHUNK_NAME="${CHUNK_NAME:-${CHUNKS[${TASK_ID}]}}"
echo "Processing chunk: ${CHUNK_NAME}"

cmd=(
  python -u scripts/extract_zip_tile_features.py
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --encoders "${ENCODERS}"
  --chunks "${CHUNK_NAME}"
  --batch-size "${BATCH_SIZE}"
  --hf-cache-dir "${HF_HOME}"
  --log-every-batches "${LOG_EVERY_BATCHES}"
)

if [[ -n "${LIMIT_ZIPS}" ]]; then
  cmd+=(--limit-zips "${LIMIT_ZIPS}")
fi

if [[ -n "${LIMIT_TILES}" ]]; then
  cmd+=(--limit-tiles "${LIMIT_TILES}")
fi

echo "Command:"
printf '  %q' "${cmd[@]}"
echo

"${cmd[@]}"

echo "---"
echo "End: $(date)"
