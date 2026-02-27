from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

OUT_DIR   = os.path.join(_HERE, "experiments", "results")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
CKPT_DIR  = os.path.join(OUT_DIR, "checkpoints")
os.makedirs(PLOTS_DIR, exist_ok=True)

def set_publication_style():
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          11,
        "axes.linewidth":     1.2,
        "axes.labelsize":     12,
        "axes.titlesize":     13,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.major.width":  1.2,
        "ytick.major.width":  1.2,
        "legend.framealpha":  0.9,
        "figure.dpi":         150,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.1,
    })


COLORS = {
    "relu":    "#1f77b4",
    "deg2":    "#ff7f0e",
    "deg3":    "#2ca02c",
    "deg4":    "#d62728",
    "int8w":   "#9467bd",
    "int8wa":  "#8c564b",
}

METHOD_COLORS = {
    "FP32_ReLU":       "#1f77b4",
    "FP32_PolyDeg2":   "#ff7f0e",
    "FP32_PolyDeg3":   "#2ca02c",
    "FP32_PolyDeg4":   "#d62728",
    "INT8W_ReLU":      "#9467bd",
    "INT8WA_ReLU":     "#8c564b",
    "INT8W_PolyDeg3":  "#e377c2",
    "INT8WA_PolyDeg3": "#7f7f7f",
}


def plot_activation_comparison(poly_coeffs_path: str,
                                 x_range: tuple = (-3, 3)):
    """
    Overlay ReLU, PolyDeg2, PolyDeg3, PolyDeg4 activations.
    Lower panel shows absolute approximation error.
    """
    from relu_poly import apply_poly_horner

    set_publication_style()
    fig, axes = plt.subplots(2, 1, figsize=(7, 6),
                              gridspec_kw={"height_ratios": [3, 1]})
    ax, ax_err = axes

    x = np.linspace(x_range[0], x_range[1], 2000)
    y_relu = np.maximum(0, x)
    ax.plot(x, y_relu, "k-", linewidth=2.5, label="ReLU (reference)", zorder=5)

    with open(poly_coeffs_path) as f:
        poly_data = json.load(f)

    styles = {"deg_2": ("--", COLORS["deg2"]),
              "deg_3": ("-.",  COLORS["deg3"]),
              "deg_4": (":",   COLORS["deg4"])}

    for key, (ls, color) in styles.items():
        if key not in poly_data:
            continue
        coeffs = np.array(poly_data[key]["coeffs"])
        y_poly = apply_poly_horner(x, coeffs)
        deg    = key.split("_")[1]
        label  = f"Poly deg {deg}  (MAE={poly_data[key]['mae']:.4f})"
        ax.plot(x, y_poly, ls, color=color, linewidth=2, label=label)

        # Error panel
        ax_err.plot(x, np.abs(y_relu - y_poly), ls, color=color, linewidth=1.5)

    ax.set_xlim(x_range)
    ax.set_ylabel("f(x)")
    ax.set_title("Activation Function: ReLU vs Polynomial Approximations")
    ax.legend(fontsize=9)
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax_err.set_xlim(x_range)
    ax_err.set_xlabel("x  (normalised current density J)")
    ax_err.set_ylabel("|error|")
    ax_err.set_yscale("log")
    ax_err.xaxis.set_minor_locator(AutoMinorLocator())

    fig.tight_layout(h_pad=0.4)
    out = os.path.join(PLOTS_DIR, "activation_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plots] Saved {out}")


def plot_scatter_all(results: pd.DataFrame,
                      y_true: np.ndarray):
    """
    2×4 grid of y_true vs y_pred scatter plots, one per experiment.
    """
    set_publication_style()
    models = list(results.index)
    n      = len(models)
    ncols  = 4
    nrows  = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for i, model_name in enumerate(models):
        ax    = axes_flat[i]
        fname = os.path.join(OUT_DIR, f"y_pred_{model_name}.csv")
        if not os.path.isfile(fname):
            ax.set_visible(False)
            continue

        y_pred = np.loadtxt(fname, delimiter=",", skiprows=1).ravel()
        color  = METHOD_COLORS.get(model_name, "#333333")

        ax.scatter(y_true, y_pred, s=10, alpha=0.5, color=color, rasterized=True)
        lo = min(y_true.min(), y_pred.min()) - 0.05
        hi = max(y_true.max(), y_pred.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("y true")
        ax.set_ylabel("y pred")
        mse_val = results.loc[model_name, "mse"]
        ax.set_title(f"{model_name}\nMSE={mse_val:.5f}", fontsize=9)
        ax.set_aspect("equal")

    # Hide unused panels
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Prediction Scatter Plots – All Methods", y=1.01, fontsize=13)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "scatter_all_methods.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plots] Saved {out}")


def plot_residual_histograms(results: pd.DataFrame, y_true: np.ndarray):
    set_publication_style()
    models = list(results.index)
    n      = len(models)
    ncols  = 4
    nrows  = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for i, model_name in enumerate(models):
        ax    = axes_flat[i]
        fname = os.path.join(OUT_DIR, f"y_pred_{model_name}.csv")
        if not os.path.isfile(fname):
            ax.set_visible(False)
            continue

        y_pred   = np.loadtxt(fname, delimiter=",", skiprows=1).ravel()
        residual = (y_true - y_pred)
        color    = METHOD_COLORS.get(model_name, "#333333")

        ax.hist(residual, bins=30, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", lw=1.2, ls="--")
        ax.set_xlabel("Residual (true − pred)")
        ax.set_ylabel("Count")
        ax.set_title(model_name, fontsize=9)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Residual Distributions – All Methods", y=1.01, fontsize=13)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "residual_histograms.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plots] Saved {out}")


def plot_weight_histograms(model_name: str = "FP32_ReLU"):
    """Show per-layer weight distributions and INT8 quantisation."""
    set_publication_style()

    import sys
    from quant import quantize_array, dequantize_array

    ckpt_path = os.path.join(CKPT_DIR, f"{model_name}.npz")
    if not os.path.isfile(ckpt_path):
        print(f"  [plots] Checkpoint not found: {ckpt_path}. Skipping.")
        return

    sd   = dict(np.load(ckpt_path))
    keys = list(sd.keys())

    fig, axes = plt.subplots(len(keys), 2,
                              figsize=(9, 3 * len(keys)))
    if len(keys) == 1:
        axes = axes[np.newaxis, :]

    for i, key in enumerate(keys):
        w    = sd[key].ravel().astype(np.float32)
        q, s = quantize_array(w.reshape(-1, 1).squeeze(), 8, signed=True)
        w_q  = dequantize_array(q, s)

        # FP32
        axes[i, 0].hist(w,   bins=50, color=COLORS["relu"],  alpha=0.8,
                          edgecolor="white", label="FP32")
        axes[i, 0].set_title(f"{key}  –  FP32", fontsize=9)
        axes[i, 0].set_ylabel("Count")

        # INT8 dequantised
        axes[i, 1].hist(w_q, bins=50, color=COLORS["int8w"], alpha=0.8,
                          edgecolor="white", label="INT8 deq")
        axes[i, 1].set_title(f"{key}  –  INT8 (dequantised)", fontsize=9)

        for ax in axes[i]:
            ax.set_xlabel("Weight value")
            ax.xaxis.set_minor_locator(AutoMinorLocator())

    fig.suptitle(f"Weight Distributions: {model_name}", fontsize=12)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "weight_histograms.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plots] Saved {out}")


def plot_experiment_bar(results: pd.DataFrame):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    models = list(results.index)
    mses   = [results.loc[m, "mse"] for m in models]
    colors = [METHOD_COLORS.get(m, "#999") for m in models]

    x = np.arange(len(models))
    bars = ax.bar(x, mses, color=colors, edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:.5f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Test MSE")
    ax.set_title("Test MSE Comparison – All Experiment Variants")
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "experiment_bar.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plots] Saved {out}")


def plot_poly_coeffs_table(poly_coeffs_path: str):
    set_publication_style()

    with open(poly_coeffs_path) as f:
        data = json.load(f)

    rows = []
    for key, val in sorted(data.items()):
        deg = int(key.split("_")[1])
        # Split coefficients onto two lines to prevent truncation
        coeffs = val["coeffs"]
        mid = (len(coeffs) + 1) // 2
        line1 = ", ".join(f"{c:.4f}" for c in coeffs[:mid])
        line2 = ", ".join(f"{c:.4f}" for c in coeffs[mid:])
        coeff_str = line1 + ("\n" + line2 if line2 else "")
        rows.append({
            "Deg":       deg,
            "MSE":       f"{val['mse']:.2e}",
            "MAE":       f"{val['mae']:.4f}",
            "Max Err":   f"{val['max_err']:.4f}",
            "HW\nAdds":  val["hw_adders"],
            "HW\nMults": val["hw_multipliers"],
            "Coefficients (a\u2080\u2026a\u2099)": coeff_str,
        })
    df = pd.DataFrame(rows)

    # Wide figure — coefficients column needs room
    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(16, 1.8 + 0.9 * n_rows))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],       # fill the whole axes
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    # Set explicit column widths — give coefficient column most space
    col_widths = [0.05, 0.09, 0.09, 0.09, 0.07, 0.07, 0.54]
    for j, w in enumerate(col_widths):
        for i in range(n_rows + 1):
            tbl[i, j].set_width(w)

    # Row height — extra for coefficient rows that may wrap
    for i in range(n_rows + 1):
        for j in range(len(df.columns)):
            tbl[i, j].set_height(0.18)

    # Header style
    for j in range(len(df.columns)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", weight="bold")

    # Alternating row shading
    for i in range(1, n_rows + 1):
        color = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(len(df.columns)):
            tbl[i, j].set_facecolor(color)

    fig.suptitle("Polynomial ReLU Approximation – Coefficient Table",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(PLOTS_DIR, "poly_coeffs_table.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [plots] Saved {out}")


def generate_all_plots(results: pd.DataFrame | None = None,
                        y_true: np.ndarray | None = None):
    """
    Generate all plots.  If results/y_true not supplied, load from disk.
    """
    poly_path = os.path.join(OUT_DIR, "poly_coeffs.json")

    if results is None:
        rpath = os.path.join(OUT_DIR, "experiment_table.csv")
        results = pd.read_csv(rpath, index_col=0)

    if y_true is None:
        # Reconstruct from first prediction file + meta
        first_pred = os.path.join(OUT_DIR, "y_pred_FP32_ReLU.csv")
        if os.path.isfile(first_pred):
            # We don't have y_true stored separately; recompute from data
            from data_prep import export_ml_csv, OUT_CSV
            if not os.path.isfile(OUT_CSV):
                export_ml_csv()
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            df  = pd.read_csv(OUT_CSV)
            y   = df["dw_out"].values.astype(np.float32)
            X   = df.drop(columns=["dw_out"]).values.astype(np.float32)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            _, _, _, y_true = train_test_split(X_scaled, y, test_size=0.2,
                                                random_state=42)

    print("\n  Generating plots …")
    if os.path.isfile(poly_path):
        plot_activation_comparison(poly_path)
        plot_poly_coeffs_table(poly_path)
    plot_scatter_all(results, y_true)
    plot_residual_histograms(results, y_true)
    plot_weight_histograms()
    plot_experiment_bar(results)
    print("  All plots done.")


if __name__ == "__main__":
    generate_all_plots()




