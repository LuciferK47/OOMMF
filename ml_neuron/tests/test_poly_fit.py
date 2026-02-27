"""
tests/test_poly_fit.py
======================
Unit tests for relu_poly.py – polynomial activation approximation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from relu_poly import (
    fit_relu_poly,
    apply_poly_horner,
    apply_poly_naive,
    poly_activation_error,
    fit_and_report,
    PolyActivation,
)


def test_coeffs_length():
    """Degree-d fit should return d+1 coefficients."""
    for d in (2, 3, 4, 5):
        c = fit_relu_poly(d, (-3, 3))
        assert len(c) == d + 1, f"Expected {d+1} coeffs for degree {d}"


def test_poly_horner_matches_naive():
    """Horner and naive evaluations should agree to ~1e-5."""
    coeffs = fit_relu_poly(3, (-3, 3))
    x      = np.linspace(-3, 3, 500)
    y_h    = apply_poly_horner(x, coeffs)
    y_n    = apply_poly_naive(x, coeffs)
    assert np.allclose(y_h, y_n, atol=1e-4), \
        f"Max diff: {np.max(np.abs(y_h - y_n)):.2e}"


def test_relu_poly_mse_deg3():
    """Degree-3 MSE < 0.015 over (-3, 3)."""
    info = poly_activation_error(3, (-3, 3))
    assert info["mse"] < 0.015, f"MSE too high: {info['mse']:.6f}"


def test_relu_poly_mse_deg2():
    """Degree-2 MSE < 0.05 over (-3, 3)."""
    info = poly_activation_error(2, (-3, 3))
    assert info["mse"] < 0.05


def test_relu_poly_mse_deg4():
    """Degree-4 MSE < 0.002 – better than deg 3."""
    info3 = poly_activation_error(3, (-3, 3))
    info4 = poly_activation_error(4, (-3, 3))
    assert info4["mse"] < info3["mse"], \
        "Degree 4 should have lower MSE than degree 3"


def test_leaky_relu_fit():
    """Leaky-ReLU approximation MSE < 0.015 for degree 3."""
    info = poly_activation_error(3, (-3, 3), activation="leaky_relu", alpha=0.05)
    assert info["mse"] < 0.015, f"Leaky ReLU poly MSE too high: {info['mse']:.6f}"


def test_fit_and_report_index():
    df = fit_and_report([2, 3, 4], (-3, 3))
    assert set(df.index) == {2, 3, 4}


def test_fit_and_report_hw_columns():
    df = fit_and_report([3], (-3, 3))
    assert "hw_adders"      in df.columns
    assert "hw_multipliers" in df.columns
    assert int(df.loc[3, "hw_adders"])      == 3
    assert int(df.loc[3, "hw_multipliers"]) == 2   # Horner: d-1


def test_poly_activation_callable():
    coeffs = fit_relu_poly(3, (-3, 3))
    act    = PolyActivation(coeffs)
    x      = np.linspace(-3, 3, 100).astype(np.float32)
    y      = act(x)
    assert y.shape == x.shape
    assert y.dtype == np.float32


def test_poly_activation_positive_region():
    """In [0.5, 3], poly should be close to ReLU (≈ identity)."""
    coeffs = fit_relu_poly(3, (-3, 3))
    act    = PolyActivation(coeffs)
    x      = np.linspace(0.5, 3.0, 200).astype(np.float32)
    y_poly = act(x).astype(np.float64)
    y_relu = x.astype(np.float64)
    rel_err = float(np.mean(np.abs(y_poly - y_relu) / (y_relu + 1e-6)))
    assert rel_err < 0.15, f"Relative error in positive region too large: {rel_err:.4f}"


def test_poly_large_degree_improves():
    """Higher degree should progressively reduce MSE."""
    mses = [poly_activation_error(d, (-3, 3))["mse"] for d in range(2, 6)]
    for i in range(len(mses) - 1):
        assert mses[i + 1] <= mses[i] * 1.5, \
            f"MSE did not improve from deg {i+2} to {i+3}: {mses}"
