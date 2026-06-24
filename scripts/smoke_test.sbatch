#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Smoke test: run the full pipeline end-to-end on one dataset,
# one language, one strategy, to verify everything works.
#
# Usage:  sbatch scripts/smoke_test.sbatch
# ─────────────────────────────────────────────────────────────────────
#SBATCH --job-name=thesis_smoke
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

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

echo "=== SMOKE TEST started at $(date) ==="
echo "GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Step 1: Build inputs for ONE dataset, ONE strategy ───────────────
echo "--- Step 1: Building judge inputs (afrimgsm, swa, zero_shot) ---"
python -m src.build_judge_inputs \
    --dataset afrimgsm \
    --lang swa \
    --strategy zero_shot

# ── Step 2: Run vLLM inference ───────────────────────────────────────
echo ""
echo "--- Step 2: Running Aya Expanse judge ---"
python -m src.run_judge_vllm \
    --model aya-expanse-32b \
    --dataset afrimgsm \
    --strategy zero_shot \
    --lang swa

# ── Step 3: Compute metrics ─────────────────────────────────────────
echo ""
echo "--- Step 3: Computing metrics ---"
python -m src.compute_metrics \
    --dataset afrimgsm \
    --judge aya-expanse-32b \
    --output "$THESIS_SCRATCH/results/smoke_test_metrics.json"

echo ""
echo "=== SMOKE TEST finished at $(date) ==="
echo ""
echo "Check results:"
echo "  Inputs:  $THESIS_SCRATCH/judge_inputs/afrimgsm/"
echo "  Outputs: $THESIS_SCRATCH/judge_outputs/afrimgsm/aya-expanse-32b/"
echo "  Metrics: $THESIS_SCRATCH/results/smoke_test_metrics.json"
