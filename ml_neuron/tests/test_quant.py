"""
tests/test_quant.py
===================
Unit tests for quant.py – INT8 symmetric quantisation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from quant import (
    quantize_array,
    dequantize_array,
    fake_quant,
    quantize_model_weights,
    dequantize_state_dict,
    QuantizedModel,
)
from neuron_fp32 import SingleNeuron


def test_quantize_output_dtype_signed():
    a = np.linspace(-1, 1, 20).astype(np.float32)
    q, s = quantize_array(a, num_bits=8, signed=True)
    assert q.dtype == np.int8, "Signed INT8 expected"
    assert isinstance(s, float)


def test_quantize_output_dtype_unsigned():
    a = np.linspace(0, 1, 20).astype(np.float32)
    q, s = quantize_array(a, num_bits=8, signed=False)
    assert q.dtype == np.uint8


def test_quantize_range_clipped():
    a = np.linspace(-1, 1, 200).astype(np.float32)
    q, s = quantize_array(a, 8, signed=True)
    assert int(q.min()) >= -128
    assert int(q.max()) <= 127


def test_dequantize_reconstruction_quality():
    """After quantise→dequantise, L∞ error ≤ scale/2 (half an LSB)."""
    a    = np.linspace(-1, 1, 256).astype(np.float32)
    q, s = quantize_array(a, 8, signed=True)
    a_hat = dequantize_array(q, s)
    max_err = float(np.max(np.abs(a - a_hat)))
    assert max_err <= s * 0.55, (
        f"Max reconstruction error {max_err:.6f} > 0.55 * scale {s:.6f}")


def test_zero_tensor_quantization():
    """All-zero tensor should not raise and scale should be 1."""
    a    = np.zeros(10, dtype=np.float32)
    q, s = quantize_array(a, 8, signed=True)
    assert s == 1.0
    assert (q == 0).all()


def test_fake_quant_shape_preserved():
    a   = np.random.randn(4, 5).astype(np.float32)
    out = fake_quant(a, 8)
    assert out.shape == a.shape


def test_quantize_model_weights_structure():
    model = SingleNeuron(in_features=4, seed=7)
    sd    = model.state_dict()
    q_sd, scales, l2_err = quantize_model_weights(sd, 8)
    assert set(q_sd.keys()) == set(sd.keys())
    assert set(scales.keys()) == set(sd.keys())
    assert l2_err >= 0.0


def test_dequantize_state_dict_dtype():
    model = SingleNeuron(in_features=3, seed=0)
    sd    = model.state_dict()
    q_sd, scales, _ = quantize_model_weights(sd, 8)
    deq_sd = dequantize_state_dict(q_sd, scales)
    for k, v in deq_sd.items():
        assert v.dtype == np.float32, f"{k} should be float32"


def test_quantized_model_output_shape():
    model  = SingleNeuron(in_features=5, seed=1)
    q_model = QuantizedModel(model, num_bits=8, quant_activations=False)
    X      = np.random.randn(10, 5).astype(np.float32)
    out    = q_model(X)
    assert out.shape == (10, 1)


def test_quantized_model_with_act_quant():
    model  = SingleNeuron(in_features=5, seed=2)
    q_model = QuantizedModel(model, num_bits=8, quant_activations=True)
    X      = np.random.randn(8, 5).astype(np.float32)
    out    = q_model(X)
    assert out.shape == (8, 1)


def test_int8_weight_error_bounded():
    """
    Weight L2 error should be small relative to original weight norm.
    For INT8 with 256 levels, relative error ≈ 1/256 ≈ 0.4 %.
    """
    model = SingleNeuron(in_features=8, seed=99)
    sd    = model.state_dict()
    q_sd, scales, l2_err = quantize_model_weights(sd, 8)

    # Original weight norm
    w_norm = float(np.sqrt(sum(np.sum(v.astype(np.float64)**2)
                               for v in sd.values())))
    rel_err = l2_err / (w_norm + 1e-9)
    assert rel_err < 0.05, (
        f"Relative INT8 weight error too large: {rel_err:.4f}")
