"""Bridge OOMMF OVF outputs to DW-neuron features and conductance values.

The module provides:
- OVF 2.0 parsing (`OMFReader`, `OMFData`)
- DW position extraction from `mz` zero-crossing
- conductance and leaky-ReLU mapping
- batch processing helpers used by `Python_omf_to_conductance.py`
"""

from __future__ import annotations

import glob
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Parsed OMF container.

@dataclass
class OMFData:
    """Structured container for a parsed OVF 2.0 file."""
    filepath:   str
    xnodes:     int
    ynodes:     int
    znodes:     int
    xstepsize:  float          # m
    ystepsize:  float          # m
    zstepsize:  float          # m
    xbase:      float          # m  (centre of first cell)
    ybase:      float          # m
    zbase:      float          # m
    # spin arrays — shape (znodes, ynodes, xnodes)
    mx: np.ndarray = field(repr=False)
    my: np.ndarray = field(repr=False)
    mz: np.ndarray = field(repr=False)

    @property
    def track_length_m(self) -> float:
        """Physical track length in metres."""
        return self.xnodes * self.xstepsize

    @property
    def x_coords_nm(self) -> np.ndarray:
        """Centre x-coordinate of each cell in nm."""
        return (self.xbase + np.arange(self.xnodes) * self.xstepsize) * 1e9


# OVF 2.0 parser.

class OMFReader:
    """
    Robust parser for OOMMF OVF 2.0 text-format files.

    Handles both text and binary OVF 2.0, though text is the format
    produced by the MIF file in this project
    (vector_field_output_format { text %.17g }).

    Usage:
        reader = OMFReader('simulation.omf')
        data   = reader.parse()
    """

    # Header key patterns (case-insensitive)
    _INT_KEYS   = {"xnodes", "ynodes", "znodes"}
    _FLOAT_KEYS = {"xstepsize", "ystepsize", "zstepsize",
                   "xbase", "ybase", "zbase",
                   "xmin", "ymin", "zmin",
                   "xmax", "ymax", "zmax"}

    def __init__(self, filepath: str):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"OMF file not found: {filepath}")
        self.filepath = filepath

    def parse(self) -> OMFData:
        """
        Parse the OVF 2.0 file and return an OMFData instance.

        Raises
        ------
        ValueError  if required header fields are missing or data section
                    is not found / has wrong number of values.
        """
        header: Dict[str, float | int] = {}
        data_lines: List[str] = []
        in_data = False

        with open(self.filepath, "r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.strip()

                # Detect data section boundaries.
                if line.lower().startswith("# begin: data text"):
                    in_data = True
                    continue
                if line.lower().startswith("# end: data text"):
                    in_data = False
                    continue

                if in_data:
                    if line and not line.startswith("#"):
                        data_lines.append(line)
                    continue

                # Parse header key/value lines.
                if not line.startswith("#"):
                    continue
                content = line.lstrip("#").strip()
                if ":" not in content:
                    continue
                key, _, val = content.partition(":")
                key = key.strip().lower()
                val = val.strip()

                if key in self._INT_KEYS:
                    try:
                        header[key] = int(val)
                    except ValueError:
                        pass
                elif key in self._FLOAT_KEYS:
                    try:
                        header[key] = float(val)
                    except ValueError:
                        pass

        # Validate required header fields.
        required = {"xnodes", "ynodes", "znodes",
                    "xstepsize", "ystepsize", "zstepsize"}
        missing = required - set(header.keys())
        if missing:
            raise ValueError(
                f"OMF header missing required fields: {missing}\n"
                f"  File: {self.filepath}"
            )

        xn = int(header["xnodes"])
        yn = int(header["ynodes"])
        zn = int(header["znodes"])
        n_cells = xn * yn * zn
        n_expected_values = n_cells * 3   # (mx, my, mz) per cell

        # Parse spin data.
        if not data_lines:
            raise ValueError(
                f"No '# Begin: Data Text' section found in {self.filepath}"
            )

        # Fast path: join all lines and split on whitespace
        raw_values = " ".join(data_lines).split()

        if len(raw_values) < n_expected_values:
            raise ValueError(
                f"Expected {n_expected_values} values ({n_cells} cells × 3) "
                f"but found {len(raw_values)} in {self.filepath}"
            )

        arr = np.array(raw_values[:n_expected_values], dtype=np.float64)

        # Reshape: OVF x-varies-fastest → (znodes, ynodes, xnodes, 3)
        arr = arr.reshape(zn, yn, xn, 3)
        mx = arr[..., 0].astype(np.float32)   # shape (zn, yn, xn)
        my = arr[..., 1].astype(np.float32)
        mz = arr[..., 2].astype(np.float32)

        return OMFData(
            filepath   = self.filepath,
            xnodes     = xn,
            ynodes     = yn,
            znodes     = zn,
            xstepsize  = float(header["xstepsize"]),
            ystepsize  = float(header["ystepsize"]),
            zstepsize  = float(header["zstepsize"]),
            xbase      = float(header.get("xbase", header["xstepsize"] / 2)),
            ybase      = float(header.get("ybase", header["ystepsize"] / 2)),
            zbase      = float(header.get("zbase", header["zstepsize"] / 2)),
            mx         = mx,
            my         = my,
            mz         = mz,
        )


# DW position extraction.

def extract_domain_wall_position(
    data: OMFData,
    layer_index: int = -1,
    saturation_threshold: float = 0.1,
) -> Tuple[Optional[float], float]:
    """
    Extract the domain-wall centre position from a parsed OMFData object.

    Method: 1D rigid-DW approximation.
      1. Select a single z-layer (default: top layer, index -1).
      2. Average mz across the transverse (y) direction → mz_avg(x).
      3. Find the x where mz_avg crosses zero (tanh-shaped DW profile).
         Sub-pixel precision via linear interpolation between the two
         bracketing cells.

    Saturation detection: if std(mz) < saturation_threshold the domain
    is uniformly magnetised (DW left the track entirely) and NaN is
    returned with a warning.

    Parameters
    ----------
    data                 : OMFData from OMFReader.parse()
    layer_index          : z-layer to use (-1 = top, 0 = bottom)
    saturation_threshold : std(mz) below this → saturated state

    Returns
    -------
    (x_dw_m, mz_std)
        x_dw_m  : DW position in metres (NaN if saturated)
        mz_std  : std(mz_avg) — diagnostic for saturation
    """
    # Select z-layer and average over y
    mz_layer = data.mz[layer_index, :, :]          # (ynodes, xnodes)
    mz_avg   = mz_layer.mean(axis=0).astype(float) # (xnodes,)
    mz_std   = float(np.std(mz_avg))

    # Saturation check
    if mz_std < saturation_threshold:
        warnings.warn(
            f"[dw_neuron_activation] Saturated state detected in "
            f"{os.path.basename(data.filepath)} "
            f"(std(mz)={mz_std:.4f} < {saturation_threshold}). "
            f"DW may have left the track. Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return np.nan, mz_std

    # Find zero-crossing: first index where sign changes
    signs = np.sign(mz_avg)
    # Replace zeros with the previous sign to avoid ambiguity
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]

    sign_changes = np.where(np.diff(signs) != 0)[0]

    if len(sign_changes) == 0:
        # No crossing — DW outside track, use track centre as fallback
        warnings.warn(
            f"[dw_neuron_activation] No mz zero-crossing found in "
            f"{os.path.basename(data.filepath)}. "
            f"Using track centre (x = L/2) as fallback.",
            UserWarning,
            stacklevel=2,
        )
        x_dw_idx = data.xnodes / 2.0
    else:
        # Use the FIRST crossing (leftmost DW if multiple exist)
        i0 = sign_changes[0]
        i1 = i0 + 1
        # Linear interpolation for sub-cell precision
        m0, m1 = float(mz_avg[i0]), float(mz_avg[i1])
        if (m1 - m0) != 0:
            frac = -m0 / (m1 - m0)
        else:
            frac = 0.5
        x_dw_idx = i0 + frac   # fractional cell index

    # Convert cell index to physical position
    x_dw_m = (data.xbase + x_dw_idx * data.xstepsize)
    return float(x_dw_m), mz_std


# Conductance mapping.

def map_dw_position_to_conductance(
    x_dw_m: float,
    track_length: float = 512e-9,
    g_min: float = 0.0,
    g_max: float = 1.0,
) -> float:
    """
    Map DW position (metres) to a raw conductance value [G_min, G_max].

    Linear bounded mapping:
        G = G_min + (G_max - G_min) * clip(x_DW / track_length, 0, 1)

    Parameters
    ----------
    x_dw_m       : DW position in metres (NaN → returns NaN)
    track_length : nanotrack length in metres (default 512 nm)
    g_min, g_max : conductance bounds

    Returns
    -------
    G_raw : float (NaN if x_dw_m is NaN)
    """
    if np.isnan(x_dw_m):
        return np.nan
    norm = float(np.clip(x_dw_m / track_length, 0.0, 1.0))
    return float(g_min + (g_max - g_min) * norm)


def leaky_relu_conductance(
    g_raw: float,
    j_density: float,
    alpha: float = 0.05,
) -> float:
    """
    Apply circuit-level leaky-ReLU to the raw conductance.

        G_act = G_raw         if J > 0   (DW driven forward)
        G_act = alpha * G_raw if J ≤ 0   (circuit asymmetry / back-current)

    The alpha·G model is a circuit-level approximation.  For physically
    accurate reverse-bias behaviour, run OOMMF with J < 0 and extract
    G(x_DW) directly from those simulations.

    Parameters
    ----------
    g_raw     : raw conductance from map_dw_position_to_conductance
    j_density : current density J in A/m² (sign determines branch)
    alpha     : leakage ratio (default 0.05, matching data_prep.py)

    Returns
    -------
    G_act : float (NaN if g_raw is NaN)
    """
    if np.isnan(g_raw):
        return np.nan
    if j_density > 0:
        return float(g_raw)
    else:
        return float(alpha * g_raw)


# Single-file processing.

def process_dw_neuron(
    omf_file: str,
    current_density: float,
    track_length: float = 512e-9,
    g_min: float = 0.0,
    g_max: float = 1.0,
    alpha: float = 0.05,
    layer_index: int = -1,
) -> Dict:
    """
    Full pipeline for a single .omf file.

    Parameters
    ----------
    omf_file        : path to OOMMF output .omf file
    current_density : J in A/m² for this simulation
    track_length    : nanotrack length in m (default 512 nm)
    g_min, g_max    : conductance bounds
    alpha           : leaky-ReLU slope for J ≤ 0
    layer_index     : z-layer (-1 = top, 0 = bottom)

    Returns
    -------
    dict with keys:
        filepath             : str
        current_density      : float  J in A/m²
        dw_position          : float  x_DW in metres (NaN if saturated)
        dw_position_nm       : float  x_DW in nm
        conductance_raw      : float  G before activation
        conductance_activated: float  G after leaky-ReLU
        mz_std               : float  diagnostic (saturation indicator)
    """
    reader      = OMFReader(omf_file)
    data        = reader.parse()
    x_dw_m, mz_std = extract_domain_wall_position(
        data, layer_index=layer_index
    )
    g_raw       = map_dw_position_to_conductance(
        x_dw_m, track_length, g_min, g_max
    )
    g_act       = leaky_relu_conductance(g_raw, current_density, alpha)

    return {
        "filepath":              omf_file,
        "current_density":       float(current_density),
        "dw_position":           float(x_dw_m) if not np.isnan(x_dw_m) else np.nan,
        "dw_position_nm":        float(x_dw_m * 1e9) if not np.isnan(x_dw_m) else np.nan,
        "conductance_raw":       float(g_raw)  if not np.isnan(g_raw)  else np.nan,
        "conductance_activated": float(g_act)  if not np.isnan(g_act)  else np.nan,
        "mz_std":                float(mz_std),
    }


# Batch processing.

def process_dw_neuron_batch(
    omf_files: List[str],
    j_array: np.ndarray,
    track_length: float = 512e-9,
    g_min: float = 0.0,
    g_max: float = 1.0,
    alpha: float = 0.05,
    layer_index: int = -1,
) -> Dict:
    """
    Batch pipeline: process one final-state .omf file per J value.

    Parameters
    ----------
    omf_files    : list of .omf file paths, one per J value (same order as j_array)
    j_array      : 1-D array of current densities in A/m² (e.g. np.arange(5,46,5)*1e10)
    track_length : nanotrack length in m
    g_min, g_max : conductance bounds
    alpha        : leaky-ReLU slope
    layer_index  : z-layer to use for DW extraction

    Returns
    -------
    dict with keys:
        conductances_activated : np.ndarray  shape (N,)  — activated G per J
        conductances_raw       : np.ndarray  shape (N,)
        dw_positions_m         : np.ndarray  shape (N,)  — x_DW in metres
        dw_positions_nm        : np.ndarray  shape (N,)  — x_DW in nm
        times                  : np.ndarray  shape (N,)  — index proxy (0…N-1)
        mz_stds                : np.ndarray  shape (N,)  — saturation diagnostics
        j_array                : np.ndarray  shape (N,)  — J values (echo)
        results_per_file       : list of dicts from process_dw_neuron()
    """
    if len(omf_files) != len(j_array):
        raise ValueError(
            f"len(omf_files)={len(omf_files)} != len(j_array)={len(j_array)}. "
            f"Provide exactly one final-state .omf file per J value."
        )

    results_per_file = []
    for omf_file, j in zip(omf_files, j_array):
        print(f"  Processing J={j/1e10:.0f}×10¹⁰ A/m²  |  {os.path.basename(omf_file)}")
        res = process_dw_neuron(
            omf_file, j, track_length, g_min, g_max, alpha, layer_index
        )
        results_per_file.append(res)

    g_act   = np.array([r["conductance_activated"] for r in results_per_file],
                        dtype=np.float64)
    g_raw   = np.array([r["conductance_raw"]       for r in results_per_file],
                        dtype=np.float64)
    x_dw_m  = np.array([r["dw_position"]           for r in results_per_file],
                        dtype=np.float64)
    x_dw_nm = np.array([r["dw_position_nm"]        for r in results_per_file],
                        dtype=np.float64)
    mz_stds = np.array([r["mz_std"]                for r in results_per_file],
                        dtype=np.float64)
    times   = np.arange(len(omf_files), dtype=np.float64)   # index proxy

    return {
        "conductances_activated": g_act,
        "conductances_raw":       g_raw,
        "dw_positions_m":         x_dw_m,
        "dw_positions_nm":        x_dw_nm,
        "times":                  times,
        "mz_stds":                mz_stds,
        "j_array":                np.asarray(j_array, dtype=np.float64),
        "results_per_file":       results_per_file,
    }


# File discovery.

def get_omf_files(directory: str = ".") -> List[str]:
    """
    Find .omf output files for the DW motion batch, returning one file
    per J-value simulation in ascending J order.

    Search strategy (in order of preference):
      1. Motion/J_*e10/ subdirectories (the canonical OOMMF_Sequential_Run.sh
         layout): within each subdirectory, take the LAST .omf file
         (alphabetically) as the final-state snapshot.
      2. Flat directory: all *.omf files in `directory` sorted alphabetically,
         EXCLUDING m_initial.omf (which is the seed, not a simulation output).

    The returned list is parallel to J_array = np.arange(5, N*5+1, 5)*1e10
    when strategy 1 is used, because J_*e10 folders are sorted numerically.

    Parameters
    ----------
    directory : path to search.  For strategy 1, this should be the repo
                root or the parent of the Motion/ folder.  For strategy 2,
                the folder containing the flat .omf files.

    Returns
    -------
    Sorted list of absolute paths to .omf files (one per J value).

    Raises
    ------
    FileNotFoundError if no .omf files are found.
    """
    # Strategy 1: Motion/J_*e10/ subdirectory layout.
    motion_base = os.path.join(directory, "Motion")
    if not os.path.isdir(motion_base):
        # Try one level up (script may be called from Motion/ itself)
        motion_base = os.path.join(directory, "..", "Motion")

    if os.path.isdir(motion_base):
        j_dirs = sorted(
            glob.glob(os.path.join(motion_base, "J_*e10")),
            key=_j_dir_sort_key,
        )
        j_dirs = [d for d in j_dirs if os.path.isdir(d)]

        found: List[str] = []
        for j_dir in j_dirs:
            omfs_in_dir = sorted(
                glob.glob(os.path.join(j_dir, "*.omf"))
            )
            # Exclude the seed file m_initial.omf
            output_omfs = [
                f for f in omfs_in_dir
                if os.path.basename(f).lower() != "m_initial.omf"
            ]
            if output_omfs:
                # Take the last file = final simulation state
                found.append(output_omfs[-1])
            else:
                warnings.warn(
                    f"[get_omf_files] No output .omf files in {j_dir} — "
                    f"has this simulation been run yet?",
                    UserWarning,
                    stacklevel=2,
                )

        if found:
            print(f"[get_omf_files] Found {len(found)} simulation output(s) "
                  f"in Motion/J_*e10/ subdirectories.")
            return found

    # Strategy 2: flat directory fallback.
    flat_omfs = sorted(
        glob.glob(os.path.join(directory, "*.omf"))
    )
    flat_omfs = [
        f for f in flat_omfs
        if os.path.basename(f).lower() != "m_initial.omf"
    ]

    if flat_omfs:
        print(f"[get_omf_files] Found {len(flat_omfs)} .omf file(s) "
              f"in flat directory: {directory}")
        return flat_omfs

    raise FileNotFoundError(
        f"No .omf output files found.\n"
        f"  Searched: {os.path.abspath(directory)}/Motion/J_*e10/  "
        f"and  {os.path.abspath(directory)}/*.omf\n"
        f"  Run OOMMF_Sequential_Run.sh first to produce simulation outputs."
    )


def _j_dir_sort_key(path: str) -> float:
    """
    Numeric sort key for Motion/J_Xe10 directory names.
    Extracts the numeric prefix, e.g. 'J_5e10' → 5.0, 'J_45e10' → 45.0.
    Falls back to lexicographic if the pattern does not match.
    """
    name = os.path.basename(path)
    m = re.match(r"J_(\d+(?:\.\d+)?)e10", name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return float("inf")   # unrecognised folders sort to the end


# Publication-style matplotlib settings.

def set_publication_style() -> None:
    """
    Apply IEEE/Nature-compliant matplotlib rcParams.

    Call once at the top of your script before any plt.figure() calls.
    """
    plt.rcParams.update({
        # Font
        "font.family":         "DejaVu Sans",
        "font.size":           11,
        # Axes
        "axes.linewidth":      1.4,
        "axes.labelsize":      12,
        "axes.titlesize":      13,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        # Ticks
        "xtick.direction":     "in",
        "ytick.direction":     "in",
        "xtick.major.width":   1.2,
        "ytick.major.width":   1.2,
        "xtick.minor.width":   0.8,
        "ytick.minor.width":   0.8,
        "xtick.major.size":    5,
        "ytick.major.size":    5,
        "xtick.minor.size":    3,
        "ytick.minor.size":    3,
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,
        # Lines
        "lines.linewidth":     2.0,
        "lines.markersize":    7,
        # Legend
        "legend.frameon":      True,
        "legend.framealpha":   0.9,
        "legend.fontsize":     10,
        "legend.edgecolor":    "0.8",
        # Figure / saving
        "figure.dpi":          150,
        "figure.figsize":      [6.4, 4.8],
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.1,
        "savefig.dpi":         300,
    })


# Self-test (run when executed directly).

if __name__ == "__main__":
    import sys

    # Locate m_initial.omf relative to this script
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    test_omf    = os.path.join(script_dir,
                               "Motion", "J_10e10", "m_initial.omf")

    if not os.path.isfile(test_omf):
        print(f"Self-test skipped: {test_omf} not found.")
        sys.exit(0)

    print("=" * 60)
    print("  dw_neuron_activation.py  –  self-test")
    print("=" * 60)

    reader = OMFReader(test_omf)
    data   = reader.parse()
    print(f"\n  Parsed:  {os.path.basename(test_omf)}")
    print(f"  Grid:    {data.xnodes} × {data.ynodes} × {data.znodes}")
    print(f"  Cell:    {data.xstepsize*1e9:.1f} × "
          f"{data.ystepsize*1e9:.1f} × {data.zstepsize*1e9:.1f} nm")

    x_dw_m, mz_std = extract_domain_wall_position(data)
    print(f"\n  DW position : {x_dw_m*1e9:.2f} nm  "
          f"(expected ≈ 112 nm for m_initial.omf)")
    print(f"  std(mz_avg) : {mz_std:.4f}")

    g_raw = map_dw_position_to_conductance(x_dw_m)
    g_act = leaky_relu_conductance(g_raw, j_density=10e10)
    print(f"\n  G_raw       : {g_raw:.6f}")
    print(f"  G_activated : {g_act:.6f}  (J=10×10¹⁰ A/m², positive → no scaling)")

    print("\n  get_omf_files() test:")
    try:
        files = get_omf_files(script_dir)
        print(f"    Found {len(files)} output file(s):")
        for f in files:
            print(f"      {f}")
    except FileNotFoundError as e:
        print(f"    (No OOMMF outputs yet — expected before simulation run)")
        print(f"    Message: {e}")

    print("\n  Self-test PASSED.\n")
