#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Full experiment run: all required strategies for one or all datasets.
#
# Usage:
#   sbatch scripts/run_experiment.sbatch                    # all datasets
#   sbatch scripts/run_experiment.sbatch flores200          # one dataset
#   sbatch scripts/run_experiment.sbatch xlsum cot          # one combo
# ─────────────────────────────────────────────────────────────────────
#SBATCH --job-name=thesis_exp
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/exp_%j.out
#SBATCH --error=logs/exp_%j.err

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

MODEL="aya-expanse-32b"
DATASET_ARG=""
STRATEGY_ARG=""
if [ -n "$1" ]; then DATASET_ARG="--dataset $1"; fi
if [ -n "$2" ]; then STRATEGY_ARG="--strategy $2"; fi

echo "=== EXPERIMENT started at $(date) ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Model:    $MODEL"
echo "Dataset:  ${1:-ALL}"
echo "Strategy: ${2:-ALL REQUIRED}"
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Step 1: Build all judge inputs ───────────────────────────────────
echo "--- Step 1: Building judge inputs ---"
python -m src.build_judge_inputs --required-only $DATASET_ARG $STRATEGY_ARG

# ── Step 2: Run vLLM inference ───────────────────────────────────────
echo ""
echo "--- Step 2: Running judge inference ---"
python -m src.run_judge_vllm --model "$MODEL" $DATASET_ARG $STRATEGY_ARG

# ── Step 3: Compute metrics ─────────────────────────────────────────
echo ""
echo "--- Step 3: Computing metrics ---"
METRICS_FILE="$THESIS_SCRATCH/results/metrics_${MODEL}_${1:-all}_$(date +%Y%m%d).json"
python -m src.compute_metrics \
    --judge "$MODEL" \
    $DATASET_ARG $STRATEGY_ARG \
    --output "$METRICS_FILE"

echo ""
echo "=== EXPERIMENT finished at $(date) ==="
echo "Results: $METRICS_FILE"
