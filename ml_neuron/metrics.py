"""Regression and quantization-related metric helpers."""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (coefficient of determination). 1.0 = perfect fit."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def relative_error_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute relative error in percent."""
    yt  = np.asarray(y_true, dtype=np.float64)
    yp  = np.asarray(y_pred, dtype=np.float64)
    eps = 1e-8 * (np.abs(yt).max() + 1.0)
    return float(100.0 * np.mean(np.abs(yt - yp) / (np.abs(yt) + eps)))


def activation_mse_vs_fp32(y_fp32: np.ndarray,
                             y_other: np.ndarray) -> float:
    """
    Activation-level MSE between a baseline FP32 prediction and
    a quantised / poly-approximated prediction.
    """
    return mse(y_fp32, y_other)


def weight_l2_error(sd_fp32: Dict[str, np.ndarray],
                     sd_other: Dict[str, np.ndarray]) -> float:
    """
    L2 norm of total weight difference:  ||W_fp32 − W_other||₂
    """
    sq_err = 0.0
    for key in sd_fp32:
        diff   = sd_fp32[key].astype(np.float64) - sd_other[key].astype(np.float64)
        sq_err += float(np.sum(diff ** 2))
    return float(np.sqrt(sq_err))


def summarise_experiment(name: str,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_fp32: np.ndarray | None = None) -> dict:
    """
    Compute all metrics for one experiment row.

    Returns a dict ready to append to an experiment results DataFrame.
    """
    row = {
        "model":          name,
        "mse":            mse(y_true, y_pred),
        "mae":            mae(y_true, y_pred),
        "rmse":           rmse(y_true, y_pred),
        "r2":             r2(y_true, y_pred),
        "rel_err_pct":    relative_error_pct(y_true, y_pred),
    }
    if y_fp32 is not None:
        row["act_mse_vs_fp32"] = activation_mse_vs_fp32(y_fp32, y_pred)
    return row


def print_table(df: pd.DataFrame, title: str = "Experiment Results") -> None:
    """Pretty-print the results table."""
    sep = "=" * 80
    print(f"\n{sep}\n{title}\n{sep}")
    float_cols = ["mse", "mae", "rmse", "r2", "rel_err_pct",
                  "act_mse_vs_fp32", "w_l2_err"]
    fmt = {}
    for c in df.columns:
        if c in float_cols:
            fmt[c] = "{:.6f}".format
    print(df.to_string(formatters=fmt))
    print(sep)
