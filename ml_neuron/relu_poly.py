"""Polynomial fitting and evaluation utilities for ReLU/Leaky-ReLU."""

from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from typing import Sequence


def fit_relu_poly(degree: int = 3,
                  x_range: tuple[float, float] = (-3.0, 3.0),
                  n_samples: int = 2000,
                  activation: str = "relu",
                  alpha: float = 0.05) -> np.ndarray:
    """
    Fit a polynomial of given degree to ReLU (or LeakyReLU) over x_range.

    Parameters
    ----------
    degree     : polynomial degree
    x_range    : (x_min, x_max) of fitting domain
    n_samples  : number of sample points
    activation : "relu" or "leaky_relu"
    alpha      : leakage slope (only used when activation="leaky_relu")

    Returns
    -------
    coeffs : ndarray shape (degree+1,) – coefficients [a0, a1, ..., a_d]
             i.e.  p(x) = a0 + a1*x + a2*x² + ...
    """
    x = np.linspace(x_range[0], x_range[1], n_samples)
    if activation == "relu":
        y = np.maximum(0.0, x)
    elif activation == "leaky_relu":
        y = np.where(x > 0, x, alpha * x)
    else:
        raise ValueError(f"Unknown activation: {activation!r}")

    p     = Polynomial.fit(x, y, deg=degree)
    p_std = p.convert()          # convert from Chebyshev to standard basis
    return p_std.coef             # [a0, a1, ..., a_degree]


def apply_poly_horner(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Evaluate polynomial using Horner's method.

        p(x) = a0 + x*(a1 + x*(a2 + x*(a3 + ...)))

    This minimises multiplications (degree−1 mults + degree adds)
    and is numerically stable.

    Parameters
    ----------
    x      : input array (float32 / float64)
    coeffs : coefficient array [a0, a1, ..., a_d]  (low → high order)

    Returns
    -------
    y : polynomial evaluated at x
    """
    x    = np.asarray(x, dtype=np.float64)
    # Reverse: Horner needs high → low order
    c    = coeffs[::-1]            # [a_d, a_{d-1}, ..., a0]
    out  = np.full_like(x, float(c[0]))
    for ci in c[1:]:
        out = out * x + float(ci)
    return out.astype(np.float32)


def apply_poly_naive(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Evaluate polynomial by summing powers (reference implementation).
    Results should match apply_poly_horner to machine precision.
    """
    x   = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    xp  = np.ones_like(x)
    for c in coeffs:
        out  += float(c) * xp
        xp   *= x
    return out.astype(np.float32)


def poly_activation_error(degree: int,
                           x_range: tuple[float, float] = (-3.0, 3.0),
                           n_samples: int = 5000,
                           activation: str = "relu",
                           alpha: float = 0.05) -> dict:
    """
    Fit polynomial of given degree and measure approximation error.

    Returns
    -------
    dict with keys: degree, mse, mae, max_err, coeffs
    """
    coeffs = fit_relu_poly(degree, x_range, n_samples=2000,
                            activation=activation, alpha=alpha)
    x = np.linspace(x_range[0], x_range[1], n_samples)
    if activation == "relu":
        y_true = np.maximum(0.0, x)
    else:
        y_true = np.where(x > 0, x, alpha * x)

    y_hat = apply_poly_horner(x, coeffs)
    err   = (y_true - y_hat).astype(np.float64)
    return {
        "degree":   degree,
        "mse":      float(np.mean(err ** 2)),
        "mae":      float(np.mean(np.abs(err))),
        "max_err":  float(np.max(np.abs(err))),
        "coeffs":   coeffs.tolist(),
    }


def fit_and_report(degrees: Sequence[int] = (2, 3, 4),
                   x_range: tuple[float, float] = (-3.0, 3.0),
                   activation: str = "relu",
                   alpha: float = 0.05) -> pd.DataFrame:
    """
    Fit polynomials of multiple degrees, return comparison DataFrame.

    Columns: degree, mse, mae, max_err, hw_adders, hw_multipliers, coeffs
    """
    rows = []
    for d in degrees:
        info = poly_activation_error(d, x_range, activation=activation, alpha=alpha)
        info["hw_adders"]      = d          # Horner: d adds
        info["hw_multipliers"] = d - 1      # Horner: d-1 mults (≥1)
        rows.append(info)
    df = pd.DataFrame(rows).set_index("degree")
    return df


def save_poly_coeffs(df: pd.DataFrame,
                     out_path: str = "experiments/results/poly_coeffs.json") -> None:
    """Serialise coefficient table to JSON."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {}
    for deg, row in df.iterrows():
        payload[f"deg_{deg}"] = {
            "mse":    row["mse"],
            "mae":    row["mae"],
            "max_err": row["max_err"],
            "coeffs": row["coeffs"] if isinstance(row["coeffs"], list)
                      else row["coeffs"].tolist(),
            "hw_adders":      int(row["hw_adders"]),
            "hw_multipliers": int(row["hw_multipliers"]),
        }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[relu_poly] Saved polynomial coefficients → {out_path}")


def load_poly_coeffs(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


class PolyActivation:
    """
    Drop-in replacement for ReLU / LeakyReLU using a polynomial approximation.

    Input is clamped to the fitting range to prevent polynomial extrapolation
    blow-up outside the training domain.

    Usage:
        coeffs = fit_relu_poly(degree=3)
        act    = PolyActivation(coeffs)
        model  = SingleNeuron(in_features=5, activation=act)
    """
    def __init__(self, coeffs: np.ndarray,
                 x_range: tuple[float, float] = (-4.0, 4.0)):
        self.coeffs  = np.asarray(coeffs, dtype=np.float64)
        self.x_range = x_range

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_clipped = np.clip(x, self.x_range[0], self.x_range[1])
        return apply_poly_horner(x_clipped, self.coeffs)

    def __repr__(self) -> str:
        deg = len(self.coeffs) - 1
        return f"PolyActivation(degree={deg}, x_range={self.x_range})"
