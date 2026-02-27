"""Prepare `dw_sim_responses.csv` from real OOMMF CSVs or physics-based synthesis."""

import os
import numpy as np
import pandas as pd

OOMMF_CSV_DIR = "../data/dw_raw_sim_outputs/"   # edit to real path if available
OUT_CSV       = os.path.join(os.path.dirname(__file__), "data", "dw_sim_responses.csv")

TRACK_LENGTH  = 512e-9   # m
G_MIN, G_MAX  = 0.0, 1.0
ALPHA_LR      = 0.05     # leaky-ReLU slope for J ≤ 0

def extract_features_from_oommf(filepath: str) -> dict:
    """
    Parse a single OOMMF output CSV and extract aggregate features.
    Expected columns: time, mz (or mx, my, mz).
    Returns a dict of feature_name → scalar value.
    """
    df = pd.read_csv(filepath)

    # Normalise column names to lower-case, strip whitespace
    df.columns = [c.strip().lower() for c in df.columns]

    if "mz" not in df.columns:
        raise ValueError(f"Column 'mz' not found in {filepath}. "
                         f"Available: {list(df.columns)}")

    mz = df["mz"].values.astype(np.float64)
    features: dict = {}
    features["mag_last"]  = float(mz[-1])
    features["mag_mean"]  = float(mz.mean())
    features["mag_std"]   = float(mz.std())
    features["mag_range"] = float(mz.max() - mz.min())

    # DW position: zero-crossing index (normalised to [0, 1])
    sign_changes = np.where(np.diff(np.sign(mz)))[0]
    if len(sign_changes) > 0:
        zc = sign_changes[0] / len(mz)
    else:
        zc = 0.5   # saturated – place at centre
    features["dw_position_norm"] = zc

    # Conductance from DW position (linear map → G_MIN … G_MAX)
    g = G_MIN + (G_MAX - G_MIN) * zc
    features["dw_out"] = float(g)
    return features


def build_dataset_from_oommf() -> pd.DataFrame:
    rows = []
    for fname in sorted(os.listdir(OOMMF_CSV_DIR)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(OOMMF_CSV_DIR, fname)
        try:
            feat = extract_features_from_oommf(path)
            rows.append(feat)
        except Exception as exc:
            print(f"  [WARN] Skipping {fname}: {exc}")
    return pd.DataFrame(rows)



def leaky_relu_conductance(J_norm: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Circuit-level leaky ReLU model (matches Python_omf_to_conductance.py)."""
    return np.where(J_norm > 0, J_norm, alpha * J_norm)


def build_synthetic_dataset(n_samples: int = 200,
                             noise_std: float = 0.015,
                             seed: int = 42) -> pd.DataFrame:
    """
    Synthesise DW neuron I/O pairs.

    Current density J ∈ [5, 45] × 10¹⁰ A/m² (matching OOMMF batch script),
    plus negative currents to model the leaky-ReLU regime.
    Each sample gets a small amount of physics-realistic Gaussian noise.
    """
    rng = np.random.default_rng(seed)

    # Positive currents (DW driven forward)
    J_pos = np.linspace(5e10, 45e10, n_samples // 2)
    # Negative currents (DW driven backward – leaky-ReLU regime)
    J_neg = np.linspace(-45e10, -5e10, n_samples // 2)
    J_all = np.concatenate([J_neg, J_pos])

    # Normalise to [−1, 1]
    J_max  = np.abs(J_all).max()
    J_norm = J_all / J_max

    # DW displacement: linear in J (rigid-wall approximation)
    # Positive J → DW moves toward +x end (higher conductance)
    dw_pos_norm = np.clip(0.5 + 0.5 * J_norm, 0.0, 1.0)

    # Conductance (linear map)
    g_raw  = G_MIN + (G_MAX - G_MIN) * dw_pos_norm

    # Leaky-ReLU activation (circuit-level)
    g_activated = np.where(J_norm > 0, g_raw, ALPHA_LR * g_raw)

    # Add noise
    noise        = rng.normal(0, noise_std, size=J_norm.shape)
    g_noisy      = np.clip(g_activated + noise, G_MIN - 0.1, G_MAX + 0.1)

    # Synthetic mz proxy (1D rigid DW: tanh profile at DW centre)
    x_track = np.linspace(0, 1, 512)
    rows = []
    for i, (jn, dw, g) in enumerate(zip(J_norm, dw_pos_norm, g_noisy)):
        mz = np.tanh(20 * (x_track - dw))           # tanh profile
        mag_last  = float(mz[-1])
        mag_mean  = float(mz.mean())
        mag_std   = float(mz.std())
        mag_range = float(mz.max() - mz.min())
        rows.append({
            "J_norm":          float(jn),
            "mag_last":        mag_last,
            "mag_mean":        mag_mean,
            "mag_std":         mag_std,
            "mag_range":       mag_range,
            "dw_position_norm": float(dw),
            "dw_out":           float(g),
        })

    return pd.DataFrame(rows)



def export_ml_csv(out_path: str = OUT_CSV,
                  oommf_dir: str | None = None) -> None:
    """
    Public API – call from existing analysis scripts to export the dataset.

    Example (add to Python_omf_to_conductance.py):
        from ml_neuron.data_prep import export_ml_csv
        export_ml_csv("ml_neuron/data/dw_sim_responses.csv")
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    src = oommf_dir or OOMMF_CSV_DIR

    if os.path.isdir(src) and any(f.endswith(".csv")
                                   for f in os.listdir(src)):
        print(f"[data_prep] Loading OOMMF CSVs from {src} ...")
        df = build_dataset_from_oommf()
    else:
        print("[data_prep] OOMMF CSVs not found – using synthetic dataset.")
        df = build_synthetic_dataset()

    df.to_csv(out_path, index=False)
    print(f"[data_prep] Wrote {len(df)} rows → {out_path}")


if __name__ == "__main__":
    export_ml_csv()
