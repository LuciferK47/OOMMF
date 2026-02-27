"""Numpy-based neuron and tiny MLP models used by the DW ML pipeline."""

from __future__ import annotations
import numpy as np


class ReLU:
    """Standard ReLU: f(x) = max(0, x)."""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)
    def __repr__(self) -> str:
        return "ReLU()"


class LeakyReLU:
    """Leaky ReLU: f(x) = x if x>0 else alpha*x. Matches DW circuit model."""
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)
    def __repr__(self) -> str:
        return f"LeakyReLU(alpha={self.alpha})"


class LinearActivation:
    """Identity / no activation."""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x
    def __repr__(self) -> str:
        return "Linear()"


class NumpyLinear:
    """
    Dense linear layer: y = x @ W.T + b
    Initialised with Kaiming uniform (He init).
    """
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, seed: int | None = None):
        rng   = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / in_features)
        self.W = rng.uniform(-limit, limit,
                              size=(out_features, in_features)).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32) if bias else None
        self.in_features  = in_features
        self.out_features = out_features

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.W.T
        if self.b is not None:
            out = out + self.b
        return out

    def state_dict(self) -> dict:
        d = {"weight": self.W.copy()}
        if self.b is not None:
            d["bias"] = self.b.copy()
        return d

    def load_state_dict(self, sd: dict) -> None:
        self.W = sd["weight"].copy().astype(np.float32)
        if "bias" in sd and self.b is not None:
            self.b = sd["bias"].copy().astype(np.float32)

    def __repr__(self) -> str:
        return (f"NumpyLinear(in={self.in_features}, "
                f"out={self.out_features}, bias={self.b is not None})")


class SingleNeuron:
    """
    Single-neuron model:  y = act( W · x + b )

    Mirrors the DW neuron: one weighted sum of inputs → conductance output.
    """
    def __init__(self, in_features: int,
                 activation=None,
                 seed: int = 0):
        self.linear = NumpyLinear(in_features, 1, seed=seed)
        self.act    = activation if activation is not None else ReLU()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.act(self.linear(x))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def state_dict(self) -> dict:
        return {"linear." + k: v
                for k, v in self.linear.state_dict().items()}

    def load_state_dict(self, sd: dict) -> None:
        lin_sd = {k.replace("linear.", "", 1): v
                  for k, v in sd.items()
                  if k.startswith("linear.")}
        self.linear.load_state_dict(lin_sd)

    def __repr__(self) -> str:
        return f"SingleNeuron(\n  {self.linear}\n  {self.act}\n)"


class TinyMLP:
    """
    Two-layer MLP:  x → Linear(in, 16) → act → Linear(16, 1)

    Provides more capacity for fitting the DW activation curve when the
    input feature space is multi-dimensional.
    """
    def __init__(self, in_features: int, hidden: int = 16,
                 activation=None, seed: int = 0):
        self.fc1  = NumpyLinear(in_features, hidden, seed=seed)
        self.fc2  = NumpyLinear(hidden, 1, seed=seed + 1)
        self.act  = activation if activation is not None else ReLU()

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self.act(self.fc1(x))
        return self.fc2(h)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def state_dict(self) -> dict:
        sd = {}
        for k, v in self.fc1.state_dict().items():
            sd[f"fc1.{k}"] = v
        for k, v in self.fc2.state_dict().items():
            sd[f"fc2.{k}"] = v
        return sd

    def load_state_dict(self, sd: dict) -> None:
        fc1_sd = {k.replace("fc1.", "", 1): v
                  for k, v in sd.items() if k.startswith("fc1.")}
        fc2_sd = {k.replace("fc2.", "", 1): v
                  for k, v in sd.items() if k.startswith("fc2.")}
        self.fc1.load_state_dict(fc1_sd)
        self.fc2.load_state_dict(fc2_sd)

    def num_params(self) -> int:
        total = 0
        for name, arr in self.state_dict().items():
            total += arr.size
        return total

    def __repr__(self) -> str:
        return (f"TinyMLP(\n  {self.fc1}\n  act={self.act}\n"
                f"  {self.fc2}\n  params={self.num_params()}\n)")


class MiniAdam:
    """
    Scalar Adam optimiser operating directly on numpy arrays via
    numerical gradients (finite differences).

    For a model with a small number of parameters this is fine for demos;
    replace with PyTorch's torch.optim.Adam for production training.
    """
    def __init__(self, model, lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8):
        self.model  = model
        self.lr     = lr
        self.beta1, self.beta2 = betas
        self.eps    = eps
        self.t      = 0

        # Initialise momentum buffers
        sd = model.state_dict()
        self.m = {k: np.zeros_like(v) for k, v in sd.items()}
        self.v = {k: np.zeros_like(v) for k, v in sd.items()}

    def zero_grad(self):
        pass   # handled inside step()

    def step(self, loss_fn, X_batch: np.ndarray, y_batch: np.ndarray,
             eps_fd: float = 1e-5):
        """
        One Adam step using central finite-difference gradients.
        """
        self.t += 1
        sd = self.model.state_dict()

        for key in sd:
            param = sd[key]
            grad  = np.zeros_like(param, dtype=np.float64)
            flat  = param.ravel()

            for idx in range(len(flat)):
                orig = flat[idx]

                flat[idx] = orig + eps_fd
                param_plus = flat.reshape(param.shape).astype(np.float32)
                sd_plus    = {**sd, key: param_plus}
                self.model.load_state_dict(sd_plus)
                loss_plus  = float(loss_fn(self.model(X_batch), y_batch))

                flat[idx] = orig - eps_fd
                param_minus = flat.reshape(param.shape).astype(np.float32)
                sd_minus    = {**sd, key: param_minus}
                self.model.load_state_dict(sd_minus)
                loss_minus  = float(loss_fn(self.model(X_batch), y_batch))

                flat[idx] = orig
                grad.ravel()[idx] = (loss_plus - loss_minus) / (2 * eps_fd)

            # Adam update
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad**2
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            sd[key] = (param - self.lr * m_hat /
                       (np.sqrt(v_hat) + self.eps)).astype(np.float32)

        self.model.load_state_dict(sd)
