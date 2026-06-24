#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Build judge input JSONL files from downloaded datasets.
# CPU-only — run after download_data.sbatch.
#
# Usage:
#   sbatch scripts/build_inputs.sbatch                          # all combos
#   sbatch scripts/build_inputs.sbatch flores200 zero_shot      # specific
# ─────────────────────────────────────────────────────────────────────
#SBATCH --job-name=thesis_build
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/build_%j.out
#SBATCH --error=logs/build_%j.err

# ── Environment ──────────────────────────────────────────────────────
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

source "$HOME/venvs/thesis_pipeline/bin/activate"
cd "$HOME/thesis_pipeline"
mkdir -p logs

export THESIS_SCRATCH="/scratch-shared/$USER"

# ── Run ──────────────────────────────────────────────────────────────
echo "=== Building judge inputs at $(date) ==="

ARGS=""
if [ -n "$1" ]; then ARGS="$ARGS --dataset $1"; fi
if [ -n "$2" ]; then ARGS="$ARGS --strategy $2"; fi

python -m src.build_judge_inputs --required-only $ARGS

echo ""
echo "=== Build finished at $(date) ==="
echo "Inputs at: $THESIS_SCRATCH/judge_inputs/"
find "$THESIS_SCRATCH/judge_inputs/" -name "*.jsonl" | wc -l
echo "JSONL files created"
