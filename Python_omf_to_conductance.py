"""Build normalized DW conductance-vs-current data from OOMMF `.omf` outputs.

Outputs:
- `data/Normalized_J.npy`
- `data/Normalized_Conductance.npy`
- `Custom_Activation_Function.png`
- `ml_neuron/data/dw_sim_responses.csv`
"""

import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

# Ensure local modules are importable when run from any working directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_NEURON  = os.path.join(_SCRIPT_DIR, "ml_neuron")

for _p in (_SCRIPT_DIR, _ML_NEURON):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dw_neuron_activation import (
    get_omf_files,
    process_dw_neuron_batch,
    set_publication_style,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert OOMMF .omf outputs to G vs J curve and .npy files."
    )
    p.add_argument(
        "--motion-dir",
        default=_SCRIPT_DIR,
        metavar="DIR",
        help=(
            "Directory to search for OOMMF outputs.  "
            "The script looks for Motion/J_*e10/ subfolders here.  "
            "Default: the directory containing this script."
        ),
    )
    p.add_argument(
        "--load-npy",
        action="store_true",
        help="Skip OOMMF parsing; load Normalized_J.npy / "
             "Normalized_Conductance.npy from data/ and re-plot.",
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join(_SCRIPT_DIR, "data"),
        metavar="DIR",
        help="Directory for .npy output files.  Default: <repo_root>/data/",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plot but do not call plt.show() (useful for headless runs).",
    )
    return p.parse_args()


TRACK_LENGTH      = 512e-9   # m — nanotrack length (xmax in MIF)
G_MIN             = 0.0
G_MAX             = 1.0
ALPHA_LEAKY_RELU  = 0.05     # circuit-level leakage ratio
LAYER_INDEX       = -1       # top z-layer (znodes=1, so -1 == 0)


def _safe_normalise(arr: np.ndarray) -> np.ndarray:
    """
    Min-max normalise an array to [0, 1].

    If all values are equal (e.g. DW pinned for every J value), the
    normalised array is all-zeros rather than NaN / ZeroDivisionError.
    """
    a_min = np.nanmin(arr)
    a_max = np.nanmax(arr)
    d_range = a_max - a_min
    if d_range < 1e-12:
        warnings.warn(
            "All conductance values are equal (range < 1e-12). "
            "This likely means the DW did not move for any J value — "
            "check that your current densities exceed the STT threshold "
            "(approximately 12×10¹⁰ A/m² for this nanotrack). "
            "Returning all-zeros normalised array.",
            UserWarning,
            stacklevel=2,
        )
        return np.zeros_like(arr)
    return (arr - a_min) / d_range


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    npy_j    = os.path.join(args.out_dir, "Normalized_J.npy")
    npy_g    = os.path.join(args.out_dir, "Normalized_Conductance.npy")
    plot_out = os.path.join(_SCRIPT_DIR, "Custom_Activation_Function.png")

    set_publication_style()

    # Load mode: reuse saved arrays and regenerate plot only.
    if args.load_npy:
        if not os.path.isfile(npy_j) or not os.path.isfile(npy_g):
            raise FileNotFoundError(
                f"--load-npy requested but files not found:\n"
                f"  {npy_j}\n  {npy_g}\n"
                f"Run without --load-npy first to generate them."
            )
        J = np.load(npy_j)
        G = np.load(npy_g)
        print(f"Loaded {len(J)} data points from .npy files.")
        _make_plot(J, G, plot_out, show=not args.no_show)
        return

    # OOMMF parsing mode.
    print("Scanning for OOMMF simulation outputs …")
    omf_files_list = get_omf_files(args.motion_dir)
    N = len(omf_files_list)
    print(f"Found {N} simulation output(s).")

    # J values follow the OOMMF batch script sequence: 5, 10, ..., N*5 ×10^10.
    J_array = np.arange(5, N * 5 + 1, 5) * 1e10   # A/m²

    print(f"Processing J = {J_array/1e10} ×10¹⁰ A/m² …\n")
    results = process_dw_neuron_batch(
        omf_files_list,
        J_array,
        TRACK_LENGTH,
        g_min=G_MIN,
        g_max=G_MAX,
        alpha=ALPHA_LEAKY_RELU,
        layer_index=LAYER_INDEX,
    )

    G_raw = results["conductances_activated"]

    # Report saturated states.
    n_nan = int(np.sum(np.isnan(G_raw)))
    if n_nan > 0:
        print(f"WARNING: {n_nan}/{N} simulation(s) returned NaN conductance "
              f"(DW likely left the track). Check mz_stds below:")
        for i, (f, s) in enumerate(
            zip(omf_files_list, results["mz_stds"])
        ):
            flag = " ← SATURATED" if np.isnan(G_raw[i]) else ""
            print(f"  J={J_array[i]/1e10:.0f}e10: std(mz)={s:.4f}  "
                  f"{os.path.basename(f)}{flag}")

    # Require at least two valid points for normalization/plotting.
    valid_mask = ~np.isnan(G_raw)
    if valid_mask.sum() < 2:
        raise ValueError(
            f"Only {valid_mask.sum()} valid (non-NaN) conductance value(s). "
            f"Cannot normalise or plot.  "
            f"Check that your OOMMF simulations completed successfully."
        )

    G = _safe_normalise(G_raw)

    J = np.arange(5, N * 5 + 1, 5, dtype=float)     # integer steps 5,10,…
    J = _safe_normalise(J)

    # Save normalized arrays.
    np.save(npy_j, J)
    np.save(npy_g, G)
    print(f"\nSaved: {npy_j}")
    print(f"Saved: {npy_g}")

    # Export CSV for the ML pipeline.
    _export_to_ml_neuron(J_array, G_raw, results)

    # Plot activation curve.
    _make_plot(J, G, plot_out, show=not args.no_show)


def _make_plot(
    J: np.ndarray,
    G: np.ndarray,
    out_path: str,
    show: bool = True,
) -> None:
    """Produce the Normalised G vs J comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(J, G, "*-", markersize=10, linewidth=3, label="OOMMF", color="#1f77b4")
    ax.plot(J, J, "k-",             linewidth=3, label="ReLU (ideal)")

    ax.set_xlabel("Normalised J")
    ax.set_ylabel("Normalised G")
    ax.legend()
    ax.tick_params(which="both", direction="in", length=6, width=2, colors="k")

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    print(f"\nSaved plot: {out_path}")

    if show:
        plt.show()
    plt.close(fig)


def _export_to_ml_neuron(
    j_array_raw: np.ndarray,
    g_activated: np.ndarray,
    results: dict,
) -> None:
    """
    Write dw_sim_responses.csv so the ML pipeline uses real OOMMF data.

    This replaces the synthetic dataset that data_prep.py falls back to
    when no OOMMF CSV directory is configured.
    """
    import pandas as pd

    ml_data_dir = os.path.join(_SCRIPT_DIR, "ml_neuron", "data")
    os.makedirs(ml_data_dir, exist_ok=True)
    out_csv = os.path.join(ml_data_dir, "dw_sim_responses.csv")

    j_max   = float(np.max(np.abs(j_array_raw)))
    j_norm  = j_array_raw / j_max

    dw_pos_norm = results["dw_positions_m"] / TRACK_LENGTH

    rows = []
    for i, (jn, dw, g) in enumerate(
        zip(j_norm, dw_pos_norm, g_activated)
    ):
        if np.isnan(g):
            continue
        rows.append({
            "J_norm":           float(jn),
            "mag_last":         float(np.nan),   # not available without time series
            "mag_mean":         float(np.nan),
            "mag_std":          float(results["mz_stds"][i]),
            "mag_range":        float(np.nan),
            "dw_position_norm": float(dw) if not np.isnan(dw) else float(np.nan),
            "dw_out":           float(g),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Exported {len(df)} rows of real OOMMF data → {out_csv}")
    print("The ML pipeline will now use real data instead of the synthetic dataset.")


if __name__ == "__main__":
    main()
