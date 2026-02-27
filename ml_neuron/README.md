# ml_neuron/

PyTorch-compatible neuron micro-framework for DW neuron ML experiments.

## What This Does

Mirrors the physical domain-wall (DW) neuron behaviour in software, then:

1. **FP32 baseline** – trains a TinyMLP with ReLU and measures accuracy
2. **INT8 quantisation** – quantises weights (and optionally activations) to 8-bit, measures accuracy degradation
3. **Polynomial ReLU** – replaces ReLU with degree-2/3/4 polynomial approximations, evaluates accuracy and hardware cost
4. **Cross comparison** – all combinations in one experiment table

## Quick Start

```bash
# From repo root
cd DW_Neuron_OOMMF_Files

# 1. Set up environment
python3 -m venv .venv_ml
source .venv_ml/bin/activate
pip install -r ml_neuron/requirements.txt

# 2. Run full pipeline
bash ml_neuron/run_experiments.sh
```

Results land in `ml_neuron/experiments/results/`.

## File Structure

```
ml_neuron/
├── data_prep.py        – generate/export dw_sim_responses.csv
├── neuron_fp32.py      – SingleNeuron, TinyMLP, ReLU, LeakyReLU
├── quant.py            – INT8 symmetric quantisation
├── relu_poly.py        – polynomial ReLU fit (degree 2–4)
├── train_and_eval.py   – full experiment loop
├── plots.py            – all publication-quality figures
├── metrics.py          – MSE, MAE, R², relative error
├── dw_mapping.md       – physics ↔ polynomial coefficient mapping
├── run_experiments.sh  – one-shot pipeline runner
├── requirements.txt
├── data/
│   └── dw_sim_responses.csv    (generated)
├── experiments/results/
│   ├── experiment_table.csv    (metrics for all variants)
│   ├── poly_coeffs.json        (polynomial coefficients)
│   ├── meta.json               (run metadata)
│   ├── y_pred_*.csv            (per-method predictions)
│   ├── checkpoints/            (saved model weights)
│   └── plots/                  (6 publication figures)
└── tests/
    ├── test_quant.py           (11 tests)
    └── test_poly_fit.py        (11 tests)
```

## Experiment Matrix

| Model            | Weights | Activations | Activation fn   |
|------------------|---------|-------------|-----------------|
| FP32_ReLU        | FP32    | FP32        | ReLU            |
| FP32_PolyDeg2    | FP32    | FP32        | Poly degree 2   |
| FP32_PolyDeg3    | FP32    | FP32        | Poly degree 3   |
| FP32_PolyDeg4    | FP32    | FP32        | Poly degree 4   |
| INT8W_ReLU       | INT8    | FP32        | ReLU            |
| INT8WA_ReLU      | INT8    | INT8        | ReLU            |
| INT8W_PolyDeg3   | INT8    | FP32        | Poly degree 3   |
| INT8WA_PolyDeg3  | INT8    | INT8        | Poly degree 3   |

## Running Without PyTorch

All code runs on **numpy + scipy + matplotlib + pandas** only.
PyTorch is listed in requirements.txt but is not required for any script to execute.
Add a `torch`-based training loop if GPU acceleration is needed.

## Integration with OOMMF Outputs

If you have real OOMMF CSV outputs, point `data_prep.py` to them:

```python
# In Python_omf_to_conductance.py (add at end):
from ml_neuron.data_prep import export_ml_csv
export_ml_csv("ml_neuron/data/dw_sim_responses.csv",
              oommf_dir="path/to/oommf/csv/outputs/")
```

Expected CSV column: `mz` (magnetisation z-component vs time/index).

## Running Tests

```bash
# With pytest:
pytest ml_neuron/tests/ -v

# Without pytest:
python ml_neuron/tests/test_quant.py
python ml_neuron/tests/test_poly_fit.py
```

## Connecting to the Viva

See `dw_mapping.md` for a detailed table connecting polynomial coefficients
to physical DW energy terms and estimated hardware implementation cost.
