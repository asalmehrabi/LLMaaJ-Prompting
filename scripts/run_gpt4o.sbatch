#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Run GPT-4o judge experiments via OpenAI API.
# CPU-only — no GPUs needed. Can run locally or on Snellius.
#
# Usage:
#   sbatch scripts/run_gpt4o.sbatch                     # all datasets
#   sbatch scripts/run_gpt4o.sbatch flores200            # one dataset
#   sbatch scripts/run_gpt4o.sbatch xlsum zero_shot      # specific combo
# ─────────────────────────────────────────────────────────────────────
#SBATCH --job-name=thesis_gpt4o
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gpt4o_%j.out
#SBATCH --error=logs/gpt4o_%j.err

# ── Environment ──────────────────────────────────────────────────────
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

source "$HOME/venvs/thesis_pipeline/bin/activate"
cd "$HOME/thesis_pipeline"
mkdir -p logs

export THESIS_SCRATCH="/scratch-shared/$USER"
# IMPORTANT: Set your OpenAI API key
export OPENAI_API_KEY=$(cat ~/.openai_key 2>/dev/null || echo '')

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set. Save key to ~/.openai_key"
    exit 1
fi

DATASET_ARG=""
STRATEGY_ARG=""
if [ -n "$1" ]; then DATASET_ARG="--dataset $1"; fi
if [ -n "$2" ]; then STRATEGY_ARG="--strategy $2"; fi

echo "=== GPT-4o EXPERIMENT started at $(date) ==="
echo "Dataset:  ${1:-ALL}"
echo "Strategy: ${2:-ALL}"

python -m src.run_judge_api --model gpt-4o $DATASET_ARG $STRATEGY_ARG

echo ""
echo "--- Computing metrics ---"
python -m src.compute_metrics --judge gpt-4o $DATASET_ARG \
    --output "$THESIS_SCRATCH/results/metrics_gpt-4o_${1:-all}_$(date +%Y%m%d).json"

echo "=== GPT-4o EXPERIMENT finished at $(date) ==="
