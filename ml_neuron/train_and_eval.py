"""Train and evaluate FP32, INT8, and polynomial-activation DW neuron models."""

from __future__ import annotations
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure local imports work from repo root or `ml_neuron/`.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from neuron_fp32  import SingleNeuron, TinyMLP, ReLU, LeakyReLU
from quant        import QuantizedModel
from relu_poly    import fit_relu_poly, PolyActivation, fit_and_report, save_poly_coeffs
from metrics      import summarise_experiment, print_table

DATA_CSV  = os.path.join(_HERE, "data", "dw_sim_responses.csv")
OUT_DIR   = os.path.join(_HERE, "experiments", "results")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
CKPT_DIR  = os.path.join(OUT_DIR, "checkpoints")

for d in (OUT_DIR, PLOTS_DIR, CKPT_DIR):
    os.makedirs(d, exist_ok=True)


def load_data(csv_path: str = DATA_CSV, target_col: str = "dw_out"):
    """
    Load dw_sim_responses.csv, split features / target, return raw arrays.
    """
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32).reshape(-1, 1)
    return X, y, feature_cols


def fit_ridge(X_train: np.ndarray, y_train: np.ndarray,
              lam: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """
    Closed-form ridge regression: W = (XᵀX + λI)⁻¹ Xᵀy
    Fast exact solution for the linear layer.
    Returns (W, b) for a single-output linear model.
    """
    N, D  = X_train.shape
    # Augment X with bias column
    ones  = np.ones((N, 1), dtype=np.float32)
    X_aug = np.hstack([X_train, ones])           # (N, D+1)
    A     = X_aug.T @ X_aug + lam * np.eye(D + 1)
    b_vec = X_aug.T @ y_train                    # (D+1, 1)
    theta = np.linalg.solve(A, b_vec)            # (D+1, 1)
    W     = theta[:-1].T                         # (1, D)
    b     = theta[-1:]                           # (1,)
    return W.astype(np.float32), b.astype(np.float32)


def build_and_train(X_train: np.ndarray, y_train: np.ndarray,
                    activation=None, seed: int = 42,
                    epochs: int = 800, lr: float = 5e-3,
                    use_mlp: bool = True) -> object:
    """
    Train SingleNeuron (or TinyMLP) with given activation.

    Strategy
    --------
    •  Initialise linear layer(s) via ridge regression on training data
       (instant, good starting point).
    •  For TinyMLP: run mini-batch stochastic gradient descent via
       analytical gradient of MSE loss w.r.t. last-layer weights,
       keeping hidden layer fixed after ridge init. This avoids torch
       dependency while being fast.

    Returns the trained model.
    """
    act  = activation if activation is not None else ReLU()
    n_in = X_train.shape[1]

    if use_mlp:
        model = TinyMLP(n_in, hidden=16, activation=act, seed=seed)
    else:
        model = SingleNeuron(n_in, activation=act, seed=seed)

    # Phase 1: ridge initialization of the last linear layer.
    # For TinyMLP: use X→act(hidden init)→ridge on output layer
    # For SingleNeuron: direct ridge
    if isinstance(model, TinyMLP):
        h  = act(X_train @ model.fc1.W.T + model.fc1.b)   # (N, 16)
        W2, b2 = fit_ridge(h, y_train)
        model.fc2.W = W2
        model.fc2.b = b2.ravel()
    else:
        W1, b1 = fit_ridge(X_train, y_train)
        model.linear.W = W1
        model.linear.b = b1.ravel()

    # Phase 2: stochastic gradient refinement (last layer only).
    batch_size = min(32, len(X_train))
    rng        = np.random.default_rng(seed)
    lr_sched   = lr

    for ep in range(epochs):
        idx      = rng.permutation(len(X_train))
        ep_loss  = 0.0
        n_batches = 0

        for start in range(0, len(X_train), batch_size):
            b_idx = idx[start:start + batch_size]
            xb    = X_train[b_idx]
            yb    = y_train[b_idx]

            if isinstance(model, TinyMLP):
                # Forward
                h   = act(xb @ model.fc1.W.T + model.fc1.b)   # (B, 16)
                h   = np.clip(h.astype(np.float64), -10.0, 10.0).astype(np.float32)
                out = h @ model.fc2.W.T + model.fc2.b          # (B, 1)
                # MSE gradient w.r.t. output layer
                err = np.clip(out - yb, -5.0, 5.0)             # (B, 1)
                dW2 = np.clip((err.T @ h) / len(xb), -1.0, 1.0)
                db2 = np.clip(err.mean(axis=0), -1.0, 1.0)
                model.fc2.W -= lr_sched * dW2
                model.fc2.b -= lr_sched * db2
            else:
                # Forward
                z   = xb @ model.linear.W.T + model.linear.b  # (B,1)
                out = act(z)
                err = out - yb
                # Gradient of ReLU-type activations: pass-through where active
                if isinstance(act, (ReLU,)):
                    da  = (z > 0).astype(np.float32)
                elif isinstance(act, LeakyReLU):
                    da  = np.where(z > 0, 1.0, act.alpha)
                else:
                    # PolyActivation: numerical derivative
                    da = np.ones_like(z)

                delta = err * da
                dW    = (delta.T @ xb) / len(xb)
                db    = delta.mean(axis=0)
                model.linear.W -= lr_sched * dW
                model.linear.b -= lr_sched * db

            ep_loss  += float(np.mean(err**2))
            n_batches += 1

        # Decay lr
        if (ep + 1) % 200 == 0:
            lr_sched *= 0.5

    return model


def save_model(model, name: str) -> str:
    path = os.path.join(CKPT_DIR, f"{name}.npz")
    np.savez(path, **model.state_dict())
    return path


def save_predictions(y_pred: np.ndarray, name: str) -> str:
    path = os.path.join(OUT_DIR, f"y_pred_{name}.csv")
    np.savetxt(path, y_pred, delimiter=",", header="y_pred", comments="")
    return path


def run_all_experiments(seed: int = 42, verbose: bool = True) -> pd.DataFrame:
    """
    Run the full experiment matrix and return a DataFrame of metrics.
    Also saves all artefacts to experiments/results/.
    """
    print("\n" + "=" * 70)
    print("  DW NEURON ML EXPERIMENTS")
    print("=" * 70)

    # 1. Data.
    X, y, feat_cols = load_data()
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X).astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, random_state=seed)

    print(f"\n  Dataset : {len(X)} samples  |  features: {feat_cols}")
    print(f"  Train   : {len(X_tr)}   Test: {len(X_te)}")

    # 2. Polynomial ReLU fits.
    # Determine input range from training activations
    x_range_data = (float(X_scaled.min()), float(X_scaled.max()))
    x_range_fit  = (max(-4, x_range_data[0]), min(4, x_range_data[1]))

    poly_df = fit_and_report(degrees=[2, 3, 4], x_range=x_range_fit)
    save_poly_coeffs(poly_df, os.path.join(OUT_DIR, "poly_coeffs.json"))

    poly_acts = {
        "PolyDeg2": PolyActivation(np.array(poly_df.loc[2, "coeffs"])),
        "PolyDeg3": PolyActivation(np.array(poly_df.loc[3, "coeffs"])),
        "PolyDeg4": PolyActivation(np.array(poly_df.loc[4, "coeffs"])),
    }

    # 3. Experiment definitions.
    experiments = [
        ("FP32_ReLU",      ReLU(),                 False, False),
        ("FP32_PolyDeg2",  poly_acts["PolyDeg2"],  False, False),
        ("FP32_PolyDeg3",  poly_acts["PolyDeg3"],  False, False),
        ("FP32_PolyDeg4",  poly_acts["PolyDeg4"],  False, False),
        ("INT8W_ReLU",     ReLU(),                 True,  False),
        ("INT8WA_ReLU",    ReLU(),                 True,  True),
        ("INT8W_PolyDeg3", poly_acts["PolyDeg3"],  True,  False),
        ("INT8WA_PolyDeg3",poly_acts["PolyDeg3"],  True,  True),
    ]

    rows      = []
    y_fp32    = None   # filled after first experiment

    for name, act, quant_w, quant_a in experiments:
        t0    = time.perf_counter()
        print(f"\n  [{name}] training ...", end=" ", flush=True)

        # Train FP32 model
        model = build_and_train(X_tr, y_tr, activation=act,
                                 seed=seed, epochs=600)

        # Wrap in quantised model if needed
        if quant_w or quant_a:
            eval_model = QuantizedModel(model, num_bits=8,
                                         quant_activations=quant_a)
            w_l2 = eval_model.weight_l2_err
        else:
            eval_model = model
            w_l2       = 0.0

        # Evaluate on test set
        y_pred = eval_model(X_te).ravel()
        y_true = y_te.ravel()

        dt = time.perf_counter() - t0
        print(f"done ({dt:.1f}s)")

        # Save predictions and checkpoint
        save_predictions(y_pred.reshape(-1, 1), name)
        save_model(model, name)

        # Metrics
        row = summarise_experiment(name, y_true, y_pred,
                                    y_fp32=y_fp32)
        row["quant_weights"] = quant_w
        row["quant_act"]     = quant_a
        row["w_l2_err"]      = w_l2
        rows.append(row)

        if name == "FP32_ReLU":
            y_fp32 = y_pred.copy()   # baseline for act-MSE comparison

    # 4. Compile results.
    results = pd.DataFrame(rows).set_index("model")
    results_path = os.path.join(OUT_DIR, "experiment_table.csv")
    results.to_csv(results_path)

    if verbose:
        print_table(results, "EXPERIMENT RESULTS")
        print(f"\n  Results saved → {results_path}")

    # 5. Save run metadata.
    meta = {
        "seed":          seed,
        "n_train":       len(X_tr),
        "n_test":        len(X_te),
        "features":      feat_cols,
        "poly_x_range":  list(x_range_fit),
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return results


if __name__ == "__main__":
    # Ensure dataset exists
    if not os.path.isfile(DATA_CSV):
        from data_prep import export_ml_csv
        export_ml_csv()

    results = run_all_experiments()
