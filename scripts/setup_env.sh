#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Setup the Python environment on Snellius for the thesis pipeline.
# Run once:  bash scripts/setup_env.sh
# ─────────────────────────────────────────────────────────────────────
set -e

echo "=== Setting up thesis pipeline environment ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# ── 1. Load Snellius modules ─────────────────────────────────────────
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

# ── 2. Create virtual environment ────────────────────────────────────
VENV_DIR="$HOME/venvs/thesis_pipeline"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR"
    python -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# ── 3. Install core packages ─────────────────────────────────────────
echo ""
echo "Installing core packages..."
pip install \
    vllm==0.6.6 \
    transformers>=4.45.0 \
    torch>=2.4.0 \
    datasets==3.5.0 \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    pyarrow

# ── 4. Install optional API packages ─────────────────────────────────
echo ""
echo "Installing API packages (for GPT-4o)..."
pip install openai

# ── 5. Create scratch directory structure ─────────────────────────────
# IMPORTANT: Change this to your own scratch path
SCRATCH="/scratch-shared/$USER"

echo ""
echo "Creating scratch directories under $SCRATCH..."
mkdir -p "$SCRATCH/raw_data"
mkdir -p "$SCRATCH/judge_inputs"
mkdir -p "$SCRATCH/judge_outputs"
mkdir -p "$SCRATCH/results"
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/hf_cache"

# ── 6. Set up environment variables ──────────────────────────────────
# Add these to your .bashrc for persistence
ENV_BLOCK="
# === Thesis pipeline environment ===
export THESIS_SCRATCH=\"$SCRATCH\"
export HF_HOME=\"$SCRATCH/hf_cache\"
export TRANSFORMERS_CACHE=\"$SCRATCH/hf_cache\"
export HF_TOKEN=\$(cat ~/.cache/huggingface/token 2>/dev/null || echo '')
"

if ! grep -q "THESIS_SCRATCH" "$HOME/.bashrc" 2>/dev/null; then
    echo "$ENV_BLOCK" >> "$HOME/.bashrc"
    echo "Environment variables added to ~/.bashrc"
else
    echo "Environment variables already in ~/.bashrc"
fi

# ── 7. Summary ──────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Scratch directories:"
echo "  $SCRATCH/raw_data        ← downloaded datasets (parquet)"
echo "  $SCRATCH/judge_inputs    ← prepared JSONL for inference"
echo "  $SCRATCH/judge_outputs   ← raw judge responses"
echo "  $SCRATCH/results         ← aggregated metrics"
echo "  $SCRATCH/logs            ← experiment logs"
echo "  $SCRATCH/hf_cache        ← HuggingFace model cache"
echo ""
echo "Next steps:"
echo "  1. Log in to HuggingFace: huggingface-cli login"
echo "  2. Download datasets:     sbatch scripts/download_data.sbatch"
echo "  3. Smoke test:            sbatch scripts/smoke_test.sbatch"
