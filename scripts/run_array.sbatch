#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# SLURM array job: run experiments across multiple datasets in parallel.
# Each array task handles one dataset with all required strategies.
#
# Usage:  sbatch scripts/run_array.sbatch
#
# The array indices map to datasets defined in DATASET_LIST below.
# Adjust the array range to match the number of datasets.
# ─────────────────────────────────────────────────────────────────────
#SBATCH --job-name=thesis_array
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-11                        # 12 datasets (adjust as needed)
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err

# ── Environment ──────────────────────────────────────────────────────
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

source "$HOME/venvs/thesis_pipeline/bin/activate"
cd "$HOME/thesis_pipeline"
mkdir -p logs

export THESIS_SCRATCH="/scratch-shared/$USER"
export HF_HOME="$THESIS_SCRATCH/hf_cache"
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo '')

# ── Dataset list (must match array range) ────────────────────────────
DATASET_LIST=(
    flores200
    mafand
    xlsum
    mlsum
    casum
    afriqa
    xquad
    persianqa
    catalanqa
    afrimgsm
    mgsm
)

# Get this task's dataset
DATASET="${DATASET_LIST[$SLURM_ARRAY_TASK_ID]}"
MODEL="aya-expanse-32b"

if [ -z "$DATASET" ]; then
    echo "No dataset for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=== ARRAY TASK $SLURM_ARRAY_TASK_ID: $DATASET ==="
echo "Job ID: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Started at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Run pipeline for this dataset ────────────────────────────────────
echo ""
echo "--- Building inputs ---"
python -m src.build_judge_inputs --dataset "$DATASET" --required-only

echo ""
echo "--- Running inference ---"
python -m src.run_judge_vllm --model "$MODEL" --dataset "$DATASET"

echo ""
echo "--- Computing metrics ---"
python -m src.compute_metrics \
    --dataset "$DATASET" --judge "$MODEL" \
    --output "$THESIS_SCRATCH/results/metrics_${DATASET}_${MODEL}.json"

echo ""
echo "=== ARRAY TASK $SLURM_ARRAY_TASK_ID finished at $(date) ==="
