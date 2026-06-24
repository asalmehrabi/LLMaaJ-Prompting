#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Download all datasets from HuggingFace and save as parquet.
# CPU-only job — no GPUs needed.
#
# Usage:
#   sbatch scripts/download_data.sbatch                 # all datasets
#   sbatch scripts/download_data.sbatch flores200       # single dataset
# ─────────────────────────────────────────────────────────────────────
#SBATCH --job-name=thesis_download
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

# ── Environment ──────────────────────────────────────────────────────
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

source "$HOME/venvs/thesis_pipeline/bin/activate"
cd "$HOME/thesis_pipeline"
mkdir -p logs

export THESIS_SCRATCH="/scratch-shared/$USER"
export HF_HOME="$THESIS_SCRATCH/hf_cache"
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo '')

# ── Run ──────────────────────────────────────────────────────────────
echo "=== Dataset download started at $(date) ==="
echo "Scratch: $THESIS_SCRATCH"

DATASET_ARG=""
if [ -n "$1" ]; then
    DATASET_ARG="--dataset $1"
    echo "Target: $1"
fi

python -m src.prepare_datasets $DATASET_ARG

echo ""
echo "=== Download finished at $(date) ==="
echo "Data saved to: $THESIS_SCRATCH/raw_data/"
ls -la "$THESIS_SCRATCH/raw_data/"
