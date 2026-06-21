#!/usr/bin/env python3
"""Inspect one current-style PURITY parquet row for geometry and truth alignment.

This script replaces the older grouped-hit notebook assumptions with the current
DetResponse ML parquet schema:

- ATAR hits come from ``atar_*`` columns.
- Time groups come from ``atar_slice_id`` / ``lyso_slice``.
- Event-level positron direction comes from ``truth_theta`` / ``truth_phi``.
- The best truth anchor is ``truth_positron_start_*`` when available.

Typical usage:

    python3 inspect_purity_parquet_alignment.py /workdir/all_ml_000.parquet --row 42

To save a figure instead of opening a window:

    python3 inspect_purity_parquet_alignment.py /workdir/all_ml_000.parquet --row 42 \
        --save /tmp/event42.png --no-show
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the parquet row inspector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet_path", type=Path, help="Path to the parquet shard to inspect.")
    parser.add_argument("--row", type=int, default=0, help="Row index to inspect.")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path where the diagnostic figure should be written.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive matplotlib window.",
    )
    return parser.parse_args()


def load_row_dict(path: Path, row_index: int) -> dict[str, object]:
    """Load a single parquet row and convert it into a plain Python dictionary."""
    table = pq.read_table(path).slice(row_index, 1)
    data = table.to_pydict()
    return {key: value[0] for key, value in data.items()}


def require_col(row: dict[str, object], name: str) -> object:
    """Return a required column value or raise a clear schema error."""
    if name not in row:
        raise KeyError(f"Missing required column '{name}'. Available keys: {sorted(row.keys())}")
    return row[name]


def optional_col(row: dict[str, object], name: str, default=None):
    """Return an optional column value with a caller-provided default."""
    return row.get(name, default)


def as_float_array(values: object) -> np.ndarray:
    """Convert a parquet list-like value into a NumPy float array."""
    return np.asarray(values if values is not None else [], dtype=float)


def as_int_array(values: object) -> np.ndarray:
    """Convert a parquet list-like value into a NumPy integer array."""
    return np.asarray(values if values is not None else [], dtype=int)


def is_finite_triplet(x: float, y: float, z: float) -> bool:
    """Check whether a 3D point is fully finite."""
    return all(np.isfinite(v) for v in (x, y, z))


def choose_truth_anchor(row: dict[str, object]) -> tuple[np.ndarray, str]:
    """Choose the most meaningful truth anchor for the positron direction overlay."""
    candidates = [
        ("truth_positron_start", "positron start"),
        ("truth_muon_stop", "muon stop"),
        ("truth_pion_stop", "pion stop"),
    ]
    for prefix, label in candidates:
        x = float(optional_col(row, f"{prefix}_x", np.nan))
        y = float(optional_col(row, f"{prefix}_y", np.nan))
        z = float(optional_col(row, f"{prefix}_z", np.nan))
        if is_finite_triplet(x, y, z):
            return np.array([x, y, z], dtype=float), label
    raise ValueError("Could not find a finite truth anchor in the row.")


def direction_from_theta_phi(theta: float, phi: float) -> np.ndarray:
    """Convert event-level positron truth angles into a 3D unit direction vector."""
    return np.array(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ],
        dtype=float,
    )


def summarize_counter(values: Iterable[int]) -> str:
    """Return a compact summary of the most common integer values."""
    counts = Counter(int(v) for v in values)
    if not counts:
        return "none"
    pieces = [f"{value}:{count}" for value, count in counts.most_common(4)]
    return ", ".join(pieces)


def print_row_summary(row: dict[str, object], atar: dict[str, np.ndarray], lyso: dict[str, np.ndarray]) -> None:
    """Print a high-level summary of the inspected parquet row."""
    anchor, anchor_label = choose_truth_anchor(row)
    theta = float(require_col(row, "truth_theta"))
    phi = float(require_col(row, "truth_phi"))
    direction = direction_from_theta_phi(theta, phi)

    print(f"schema_version: {optional_col(row, 'schema_version', 'UNKNOWN')}")
    print(f"event_id: {optional_col(row, 'event_id', 'UNKNOWN')}")
    print(f"ATAR hits: {len(atar['x'])}")
    print(f"LYSO hits: {len(lyso['x'])}")
    print(f"anchor: {anchor_label} @ ({anchor[0]:.3f}, {anchor[1]:.3f}, {anchor[2]:.3f})")
    print(f"truth direction: ({direction[0]:.5f}, {direction[1]:.5f}, {direction[2]:.5f})")
    print(f"truth positron energy: {float(optional_col(row, 'truth_positron_energy', np.nan)):.5f}")


def print_slice_summary(atar: dict[str, np.ndarray], lyso: dict[str, np.ndarray]) -> None:
    """Print per-slice diagnostics for ATAR and LYSO hits."""
    if len(atar["slice_id"]) == 0:
        print("No ATAR slice information found.")
        return

    unique_slices = sorted(int(v) for v in np.unique(atar["slice_id"]))
    lyso_by_slice: dict[int, np.ndarray] = defaultdict(lambda: np.array([], dtype=int))
    if len(lyso["slice_id"]) > 0:
        for slice_id in sorted(int(v) for v in np.unique(lyso["slice_id"])):
            lyso_by_slice[slice_id] = np.where(lyso["slice_id"] == slice_id)[0]

    print("\nPer-slice summary:")
    for slice_id in unique_slices:
        hit_idx = np.where(atar["slice_id"] == slice_id)[0]
        x_hits = int(np.sum(atar["view"][hit_idx] == 0))
        y_hits = int(np.sum(atar["view"][hit_idx] == 1))
        lyso_idx = lyso_by_slice.get(slice_id, np.array([], dtype=int))
        time_min = float(np.min(atar["t"][hit_idx])) if len(hit_idx) else float("nan")
        time_max = float(np.max(atar["t"][hit_idx])) if len(hit_idx) else float("nan")
        mean_t = float(atar["slice_mean_t"][hit_idx[0]]) if len(hit_idx) else float("nan")

        print(
            f"  slice {slice_id:2d}: "
            f"atar_hits={len(hit_idx):3d} "
            f"(x={x_hits:3d}, y={y_hits:3d}) "
            f"lyso_hits={len(lyso_idx):2d} "
            f"time=[{time_min:8.3f}, {time_max:8.3f}] "
            f"mean_t={mean_t:8.3f} "
            f"origin={summarize_counter(atar['origin'][hit_idx])} "
            f"pdg={summarize_counter(atar['pdg'][hit_idx])}"
        )


def extract_detector_payload(row: dict[str, object], prefix: str) -> dict[str, np.ndarray]:
    """Extract the current parquet columns for one detector subsystem."""
    payload: dict[str, np.ndarray] = {
        "x": as_float_array(require_col(row, f"{prefix}_x")),
        "y": as_float_array(require_col(row, f"{prefix}_y")),
        "z": as_float_array(require_col(row, f"{prefix}_z")),
        "t": as_float_array(require_col(row, f"{prefix}_t")),
        "e": as_float_array(require_col(row, f"{prefix}_E")),
        "pdg": as_int_array(optional_col(row, f"{prefix}_pdg", [])),
        "origin": as_int_array(optional_col(row, f"{prefix}_origin", [])),
    }

    if prefix == "atar":
        payload["view"] = as_int_array(require_col(row, "atar_view"))
        payload["slice_id"] = as_int_array(require_col(row, "atar_slice_id"))
        payload["slice_mean_t"] = as_float_array(require_col(row, "atar_slice_mean_t"))
        payload["truth_t"] = as_float_array(optional_col(row, "atar_truth_t", []))
    else:
        payload["slice_id"] = as_int_array(optional_col(row, "lyso_slice", []))
        payload["slice_mean_t"] = as_float_array(optional_col(row, "lyso_slice_mean_t", []))
    return payload


def plot_alignment(
    row: dict[str, object],
    parquet_path: Path,
    row_index: int,
    atar: dict[str, np.ndarray],
    lyso: dict[str, np.ndarray],
) -> plt.Figure:
    """Build a multi-panel diagnostic plot for one parquet row."""
    anchor, anchor_label = choose_truth_anchor(row)
    direction = direction_from_theta_phi(float(require_col(row, "truth_theta")), float(require_col(row, "truth_phi")))
    unique_slices = sorted(int(v) for v in np.unique(atar["slice_id"])) if len(atar["slice_id"]) else [0]
    cmap = plt.get_cmap("tab10", max(1, len(unique_slices)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_xz, ax_yz, ax_tz = axes

    z_span = float(np.ptp(atar["z"])) if len(atar["z"]) else 1.0
    arrow_scale = max(2.0, 0.2 * z_span)

    for color_index, slice_id in enumerate(unique_slices):
        color = cmap(color_index)
        mask = atar["slice_id"] == slice_id
        x_mask = mask & (atar["view"] == 0)
        y_mask = mask & (atar["view"] == 1)

        if np.any(x_mask):
            ax_xz.scatter(
                atar["x"][x_mask],
                atar["z"][x_mask],
                s=22,
                alpha=0.85,
                color=color,
                label=f"slice {slice_id}",
            )
        if np.any(y_mask):
            ax_yz.scatter(
                atar["y"][y_mask],
                atar["z"][y_mask],
                s=22,
                alpha=0.85,
                color=color,
                label=f"slice {slice_id}",
            )
        if np.any(mask):
            ax_tz.scatter(
                atar["t"][mask],
                atar["z"][mask],
                s=18,
                alpha=0.8,
                color=color,
                label=f"slice {slice_id}",
            )

    if len(lyso["slice_id"]) > 0 and len(lyso["x"]) > 0:
        ax_tz.scatter(
            lyso["t"],
            lyso["z"],
            s=np.clip(lyso["e"] * 4.0, 12.0, 80.0),
            alpha=0.55,
            marker="x",
            color="black",
            label="LYSO",
        )

    ax_xz.arrow(
        anchor[0],
        anchor[2],
        direction[0] * arrow_scale,
        direction[2] * arrow_scale,
        width=0.0,
        head_width=0.10 * arrow_scale,
        head_length=0.12 * arrow_scale,
        color="black",
        alpha=0.9,
        length_includes_head=True,
    )
    ax_yz.arrow(
        anchor[1],
        anchor[2],
        direction[1] * arrow_scale,
        direction[2] * arrow_scale,
        width=0.0,
        head_width=0.10 * arrow_scale,
        head_length=0.12 * arrow_scale,
        color="black",
        alpha=0.9,
        length_includes_head=True,
    )

    ax_xz.set_title(f"ATAR XZ by slice\nanchor={anchor_label}")
    ax_xz.set_xlabel("x [mm]")
    ax_xz.set_ylabel("z [mm]")

    ax_yz.set_title("ATAR YZ by slice")
    ax_yz.set_xlabel("y [mm]")
    ax_yz.set_ylabel("z [mm]")

    ax_tz.set_title("ATAR/LYSO time vs z")
    ax_tz.set_xlabel("t [ns]")
    ax_tz.set_ylabel("z [mm]")

    for axis in axes:
        axis.grid(alpha=0.25)

    handles, labels = ax_xz.get_legend_handles_labels()
    handles_tz, labels_tz = ax_tz.get_legend_handles_labels()
    dedup: dict[str, object] = {}
    for handle, label in list(zip(handles, labels)) + list(zip(handles_tz, labels_tz)):
        dedup[label] = handle
    if dedup:
        fig.legend(dedup.values(), dedup.keys(), loc="upper center", ncol=min(8, len(dedup)))

    fig.suptitle(f"Row {row_index} | {parquet_path.name}", y=1.02)
    fig.tight_layout()
    return fig


def main() -> None:
    """Run the parquet row alignment inspector."""
    args = parse_args()
    row = load_row_dict(args.parquet_path, args.row)
    atar = extract_detector_payload(row, "atar")
    lyso = extract_detector_payload(row, "lyso")

    print(f"Loaded row {args.row} from {args.parquet_path}")
    print(f"Columns: {len(row)}")
    print_row_summary(row, atar, lyso)
    print_slice_summary(atar, lyso)

    figure = plot_alignment(row, args.parquet_path, args.row, atar, lyso)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=160, bbox_inches="tight")
        print(f"\nSaved figure to {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
