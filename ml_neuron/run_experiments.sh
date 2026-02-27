#!/bin/bash
# Run the full DW neuron ML pipeline (data prep, tests, training, plots).

set -e

# Always run from `ml_neuron/`.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo " DW Neuron ML Experiment Pipeline"
echo " Working dir: $(pwd)"
echo "================================================"
echo ""

# Optional virtual environment activation.
if [ -f "../.venv_ml/bin/activate" ]; then
    source "../.venv_ml/bin/activate"
    echo "[INFO] Activated virtual environment: ../.venv_ml"
fi

echo "[1/4] Preparing dataset..."
python data_prep.py
echo ""

echo "[2/4] Running unit tests..."
if command -v pytest &> /dev/null; then
    pytest tests/ -v --tb=short
else
    python -m pytest tests/ -v --tb=short
fi
echo ""

echo "[3/4] Running all experiments (FP32, INT8, PolyReLU)..."
python train_and_eval.py
echo ""

echo "[4/4] Generating plots..."
python plots.py
echo ""

echo "================================================"
echo " Pipeline complete!"
echo " Results: ml_neuron/experiments/results/"
echo " Plots  : ml_neuron/experiments/results/plots/"
echo "================================================"
