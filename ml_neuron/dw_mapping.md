# DW Neuron Physics ↔ Polynomial Activation Mapping

## Overview

The domain-wall (DW) neuron in this repo maps spin current density J to a
conductance output G through three physical stages:

```
J (input) → DW displacement x_DW(J) → G(x_DW) → Activation
```

Each stage has a direct analogue in the polynomial approximation of ReLU.

---

## Stage-by-Stage Mapping

### 1. Input: Spin Current Density J

| Physics                     | ML neuron                        |
|-----------------------------|----------------------------------|
| J  (A/m²)                   | Normalised input  x ∈ [−1, 1]   |
| Positive J → DW moves +x    | x > 0  (active ReLU region)      |
| Negative J → DW pinned/back | x ≤ 0  (leaky / suppressed)      |

**Normalisation used:**  x = J / J_max,  J_max = 45 × 10¹⁰ A/m²

---

### 2. DW Displacement: Linear Response Regime

For the 512 nm nanotrack with parameters A=15 pJ/m, D=2 mJ/m², Ms=1.1 MA/m, Ku=1 MJ/m³
the steady-state DW position scales approximately linearly with J (Walker
regime before breakdown):

```
x_DW(J) ≈ L/2 + k·J        (k ≈ 0.5 nm per MA/cm²)
```

In the polynomial activation the **linear term a₁·x** captures exactly this
linear DW response. The coefficient **a₁ ≈ 0.48** in the degree-3 fit
corresponds to the normalised slope of the DW transfer curve.

---

### 3. Conductance Mapping: Linear Saturation

```
G(x_DW) = G_min + (G_max - G_min) · (x_DW / L)
```

This is a bounded linear map — directly reproduced by the **a₀ + a₁·x** terms.

- **a₀** (constant offset) ↔ G_min (conductance floor, ≈ physical contact resistance)
- **a₁** (linear slope)   ↔ dG/dx_DW × L (magnetoresistance coefficient)

---

### 4. Leaky-ReLU Activation

```
f(J) = G(x_DW)      if J > 0     [full DW motion, read-out enabled]
f(J) = α · G(x_DW)  if J ≤ 0    [circuit-level asymmetry, α = 0.05]
```

The leakage factor **α = 0.05** represents the **back-current ratio** of the
spin-torque readout bridge, not an intrinsic material property.

In the polynomial, the negative-side suppression is captured by the **even-order
terms (a₀, a₂, a₄…)** which create a smooth kink near x = 0, avoiding the
non-differentiable corner of hard ReLU.

---

## Polynomial Coefficients vs Physical Interpretation

| Coefficient | Symbol | Physical analogue | HW resource |
|-------------|--------|-------------------|-------------|
| a₀          | offset | G_min (contact resistance) | 1 constant (no op) |
| a₁          | linear | dG/dx_DW (magnetoresistance) | 1 adder |
| a₂          | quadratic | Walker breakdown onset curvature | 1 multiplier + 1 adder |
| a₃          | cubic  | Saturation at high J | 1 multiplier + 1 adder |
| a₄          | quartic | Fine curvature near x=0 | 1 multiplier + 1 adder |

**Horner's method** (`p(x) = a₀ + x(a₁ + x(a₂ + …))`) reduces hardware cost:
- Degree d requires **d multiplications** and **d additions** total.
- Each coefficient requires ~8-bit fixed-point precision for INT8 arithmetic.

---

## Hardware Cost Table (Horner, INT8 coefficients)

| Degree | Mults | Adds | Coeff bits (total) | Approx area (FO4 gates) | Act. MSE vs ReLU |
|--------|-------|------|--------------------|-------------------------|-----------------|
| 2      | 1     | 2    | 24                 | ~50                     | ~0.012          |
| 3      | 2     | 3    | 32                 | ~80                     | ~0.003          |
| 4      | 3     | 4    | 40                 | ~115                    | ~0.0008         |

**Recommended:** Degree 3 offers the best hardware-accuracy tradeoff.
The third-order term captures the smooth onset of DW motion without requiring
a hard comparator (as ReLU would need in digital circuits).

---

## DW-Specific Notes for Viva

1. **Why not exact ReLU?**  The physical activation is smooth (not piecewise
   linear) because DW motion starts gradually as J increases from zero (Walker
   regime). A polynomial of degree ≥ 3 naturally captures this smooth onset.

2. **INT8 compatibility:**  The DW nanotrack operates with a continuous
   conductance range [G_min, G_max]. Mapping this to 256 levels (INT8) gives
   a resolution of ΔG = (G_max − G_min)/255 ≈ 0.004 S — well within
   measurement noise of typical GMR/TMR readout.

3. **Negative current regime:**  Real OOMMF simulations with J < 0 show DW
   motion in the −x direction (smaller G). The α·G approximation is valid when
   the readout circuit provides negligible backward drive; for accurate
   bi-directional modelling, run OOMMF with signed J and extract G(x_DW)
   directly.

4. **Energy landscape:**  The cubic coefficient a₃ is related to the
   anharmonicity of the DW energy potential well. In experimental devices,
   a₃ > 0 corresponds to a softening potential (DW displaced easily at high J),
   consistent with the observed super-linear G(J) at large current densities.
