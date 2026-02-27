"""Symmetric INT8 quantization utilities for arrays and model state dicts."""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any


def quantize_array(x: np.ndarray,
                   num_bits: int = 8,
                   signed: bool = True) -> Tuple[np.ndarray, float]:
    """
    Quantise a float32 array to INT8 (or UINT8) with symmetric per-tensor scale.

    Parameters
    ----------
    x        : float32 ndarray
    num_bits : bit-width (default 8)
    signed   : INT8 if True, UINT8 if False

    Returns
    -------
    q     : quantised integer array (int8 or uint8)
    scale : float32 scale factor  (x ≈ q * scale)
    """
    x = x.astype(np.float32)
    if signed:
        q_min, q_max = -(2 ** (num_bits - 1)), (2 ** (num_bits - 1)) - 1
        dtype = np.int8
    else:
        q_min, q_max = 0, (2 ** num_bits) - 1
        dtype = np.uint8

    max_abs = float(np.max(np.abs(x)))
    if max_abs == 0.0:
        scale = 1.0
    else:
        scale = max_abs / q_max

    q = np.clip(np.round(x / scale), q_min, q_max).astype(dtype)
    return q, scale


def dequantize_array(q: np.ndarray, scale: float) -> np.ndarray:
    """
    Reconstruct float32 from quantised integer array.

    Parameters
    ----------
    q     : int8 / uint8 ndarray
    scale : scale factor returned by quantize_array

    Returns
    -------
    x_hat : float32 reconstruction  (= q * scale)
    """
    return q.astype(np.float32) * scale


def fake_quant(x: np.ndarray,
               num_bits: int = 8,
               signed: bool = True) -> np.ndarray:
    """
    Round-trip quantise→dequantise (simulates quantisation noise in FP32).
    Used for activation quantisation without leaving the FP32 compute graph.
    """
    q, scale = quantize_array(x, num_bits, signed)
    return dequantize_array(q, scale)


def quantize_model_weights(state_dict: Dict[str, np.ndarray],
                            num_bits: int = 8
                            ) -> Tuple[Dict[str, np.ndarray],
                                       Dict[str, float],
                                       float]:
    """
    Quantise all weight/bias arrays in a model state_dict.

    Returns
    -------
    q_sd     : dict of INT8 arrays
    scales   : dict of per-tensor scale factors
    w2_error : L2 weight quantisation error (||W - W_hat||₂)
    """
    q_sd: Dict[str, np.ndarray] = {}
    scales: Dict[str, float]    = {}
    sq_err = 0.0
    n_elem = 0

    for key, val in state_dict.items():
        q, s           = quantize_array(val, num_bits, signed=True)
        q_sd[key]      = q
        scales[key]    = s
        deq            = dequantize_array(q, s)
        sq_err        += float(np.sum((val.astype(np.float32) - deq) ** 2))
        n_elem        += val.size

    w2_error = float(np.sqrt(sq_err))
    return q_sd, scales, w2_error


def dequantize_state_dict(q_sd: Dict[str, np.ndarray],
                           scales: Dict[str, float]
                           ) -> Dict[str, np.ndarray]:
    """Reconstruct float32 state_dict from INT8 state_dict + scales."""
    return {k: dequantize_array(v, scales[k]) for k, v in q_sd.items()}


class QuantizedModel:
    """
    Wraps any model that has a state_dict / load_state_dict interface.

    On construction, quantises all weights.
    On __call__, dequantises weights to FP32, runs forward pass, and
    optionally quantises activations too (fake_quant).

    This emulates the error introduced by INT8 inference while running in FP32.
    """

    def __init__(self, fp32_model: Any,
                 num_bits: int = 8,
                 quant_activations: bool = False):
        import copy
        self._model           = copy.deepcopy(fp32_model)
        self.num_bits         = num_bits
        self.quant_activations = quant_activations

        # Quantise weights at construction time
        sd                         = fp32_model.state_dict()
        self._q_sd, self._scales, self.weight_l2_err = \
            quantize_model_weights(sd, num_bits)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Dequantise weights → load into FP32 model
        deq_sd = dequantize_state_dict(self._q_sd, self._scales)
        self._model.load_state_dict(deq_sd)

        # Forward pass
        out = self._model(x)

        # Optionally quantise activations (simulates INT8 accumulate)
        if self.quant_activations:
            out = fake_quant(out, self.num_bits)
        return out

    def weight_scales_summary(self) -> dict:
        """Return per-layer scale factors and saturation info."""
        summary = {}
        for key, q in self._q_sd.items():
            s  = self._scales[key]
            summary[key] = {
                "scale":       s,
                "q_min":       int(q.min()),
                "q_max":       int(q.max()),
                "utilisation": float((q.max() - q.min()) / 255),
            }
        return summary


def weight_histograms(state_dict: Dict[str, np.ndarray],
                      bins: int = 64) -> Dict[str, Tuple]:
    """
    Return histogram (counts, edges) for each weight tensor.
    Useful for visualising quantisation clipping.
    """
    hists = {}
    for key, val in state_dict.items():
        flat = val.ravel().astype(np.float32)
        counts, edges = np.histogram(flat, bins=bins)
        hists[key] = (counts, edges)
    return hists
