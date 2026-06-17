from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from html import escape
from pathlib import Path
import math

from IPython.display import HTML, display
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


DEFAULT_PARQUET_PATH = Path("/home/jack/python_projects/pioneerML/data/purity_inputs/all_ml_000.parquet")
DEFAULT_ROW_INDEX = 0
FIG_DPI = 120
SPATIAL_FIGSIZE = (8.6, 5.2)
TIME_FIGSIZE = (8.6, 4.8)
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


plt.rcParams["figure.dpi"] = FIG_DPI
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25


@dataclass(frozen=True)
class EventData:
    row_index: int
    row: dict
    truth: dict
    atar: dict
    lyso: dict


@lru_cache(maxsize=4)
def load_table(path_str):
    return pq.read_table(Path(path_str))


def load_row_dict(path: Path, row_index: int):
    table = load_table(str(path)).slice(row_index, 1)
    data = table.to_pydict()
    return {key: values[0] for key, values in data.items()}


def optional_col(row, name, default=None):
    return row.get(name, default)


def require_col(row, name):
    if name not in row:
        raise KeyError(f"Missing required column '{name}'. Available keys: {sorted(row.keys())}")
    return row[name]


def as_float_array(values):
    return np.asarray(values if values is not None else [], dtype=float)


def as_int_array(values):
    return np.asarray(values if values is not None else [], dtype=int)


def finite_triplet(values):
    arr = np.asarray(values, dtype=float)
    return arr.shape == (3,) and np.all(np.isfinite(arr))


def point_from_prefix(row, prefix):
    return np.array(
        [
            float(optional_col(row, f"{prefix}_x", np.nan)),
            float(optional_col(row, f"{prefix}_y", np.nan)),
            float(optional_col(row, f"{prefix}_z", np.nan)),
        ],
        dtype=float,
    )


def direction_from_theta_phi(theta, phi):
    return np.array(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ],
        dtype=float,
    )


def extract_detector_payload(row, prefix):
    payload = {
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


def build_truth_payload(row):
    truth = {
        "theta": float(optional_col(row, "truth_theta", np.nan)),
        "phi": float(optional_col(row, "truth_phi", np.nan)),
        "positron_energy": float(optional_col(row, "truth_positron_energy", np.nan)),
        "pion_start": point_from_prefix(row, "truth_pion_start"),
        "pion_stop": point_from_prefix(row, "truth_pion_stop"),
        "muon_start": point_from_prefix(row, "truth_muon_start"),
        "muon_stop": point_from_prefix(row, "truth_muon_stop"),
        "positron_start": point_from_prefix(row, "truth_positron_start"),
        "positron_stop": point_from_prefix(row, "truth_positron_stop"),
    }
    if np.isfinite(truth["theta"]) and np.isfinite(truth["phi"]):
        truth["direction"] = direction_from_theta_phi(truth["theta"], truth["phi"])
    else:
        truth["direction"] = np.array([np.nan, np.nan, np.nan], dtype=float)
    return truth


def segment_length(start, stop):
    if not (finite_triplet(start) and finite_triplet(stop)):
        return np.nan
    return float(np.linalg.norm(stop - start))


def pdg_color_map(pdgs):
    unique_pdgs = sorted(set(int(v) for v in np.asarray(pdgs, dtype=int).tolist()))
    base_colors = {
        -11: "tab:cyan",
        11: "tab:blue",
        -13: "tab:orange",
        13: "tab:red",
        -211: "tab:olive",
        211: "tab:green",
        22: "goldenrod",
    }
    fallback = plt.get_cmap("tab10", max(1, len(unique_pdgs)))
    color_by_pdg = {}
    fallback_index = 0
    for pdg in unique_pdgs:
        if pdg in base_colors:
            color_by_pdg[pdg] = base_colors[pdg]
        else:
            color_by_pdg[pdg] = fallback(fallback_index)
            fallback_index += 1
    return color_by_pdg


def scatter_hits_by_pdg(ax, xvals, yvals, pdgs, base_label, marker="o", size=18, alpha=0.85, size_scale=None):
    if len(xvals) == 0:
        return

    if len(pdgs) != len(xvals):
        sizes = size_scale if size_scale is not None else size
        ax.scatter(xvals, yvals, s=sizes, alpha=alpha, marker=marker, color="tab:gray", label=base_label)
        return

    pdgs = np.asarray(pdgs, dtype=int)
    color_by_pdg = pdg_color_map(pdgs)
    for pdg in sorted(color_by_pdg):
        mask = pdgs == pdg
        if not np.any(mask):
            continue
        sizes = size_scale[mask] if size_scale is not None else size
        ax.scatter(
            xvals[mask],
            yvals[mask],
            s=sizes,
            alpha=alpha,
            marker=marker,
            color=color_by_pdg[pdg],
            label=f"{base_label} pdg={pdg}",
        )


def scatter_truth_point(ax, point, axis_pair, color, label, marker="o", size=70, alpha=0.85, zorder=5):
    if not finite_triplet(point):
        return
    i0 = AXIS_INDEX[axis_pair[0]]
    i1 = AXIS_INDEX[axis_pair[1]]
    ax.scatter(
        [point[i0]],
        [point[i1]],
        color=color,
        label=label,
        marker=marker,
        s=size,
        alpha=alpha,
        zorder=zorder,
    )


def collect_xyz_points(atar, lyso, truth, include_upstream=False, local=False):
    points = []

    if len(atar["x"]) > 0:
        points.append(np.column_stack([atar["x"], atar["y"], atar["z"]]))
    if not local and len(lyso["x"]) > 0:
        points.append(np.column_stack([lyso["x"], lyso["y"], lyso["z"]]))

    truth_names = ["pion_stop", "muon_start", "muon_stop", "positron_start", "positron_stop"]
    if include_upstream or not local:
        truth_names.insert(0, "pion_start")

    for name in truth_names:
        point = truth[name]
        if finite_triplet(point):
            points.append(point.reshape(1, 3))

    if points:
        return np.vstack(points)
    return np.empty((0, 3), dtype=float)


def compute_limits(points, axis_pair, margin_fraction=0.15, min_span=8.0):
    if points.size == 0:
        return (-10.0, 10.0), (-10.0, 10.0)
    i0 = AXIS_INDEX[axis_pair[0]]
    i1 = AXIS_INDEX[axis_pair[1]]
    xvals = points[:, i0]
    yvals = points[:, i1]
    xmin, xmax = float(np.min(xvals)), float(np.max(xvals))
    ymin, ymax = float(np.min(yvals)), float(np.max(yvals))
    xspan = max(xmax - xmin, min_span)
    yspan = max(ymax - ymin, min_span)
    xpad = margin_fraction * xspan
    ypad = margin_fraction * yspan
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    return (
        xmid - 0.5 * xspan - xpad,
        xmid + 0.5 * xspan + xpad,
    ), (
        ymid - 0.5 * yspan - ypad,
        ymid + 0.5 * yspan + ypad,
    )


def compute_bound_with_padding(values, min_pad=0.15, pad_fraction=0.02):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (-10.0, 10.0)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    span = max(hi - lo, 1.0)
    pad = max(min_pad, pad_fraction * span)
    return lo - pad, hi + pad


def clamp_interval(interval, bounds):
    lo, hi = float(interval[0]), float(interval[1])
    bound_lo, bound_hi = float(bounds[0]), float(bounds[1])
    width = hi - lo
    if width >= (bound_hi - bound_lo):
        return bound_lo, bound_hi
    lo = max(lo, bound_lo)
    hi = min(hi, bound_hi)
    if hi - lo < width:
        if lo <= bound_lo:
            hi = min(bound_hi, bound_lo + width)
        else:
            lo = max(bound_lo, bound_hi - width)
    return lo, hi


def flatten_ragged_float_column(column):
    arrays = [np.asarray(values, dtype=float) for values in column.to_pylist() if values is not None and len(values)]
    if not arrays:
        return np.empty(0, dtype=float)
    return np.concatenate(arrays)


@lru_cache(maxsize=8)
def load_atar_section_bounds(path_str):
    table = load_table(path_str)
    return {
        "x": compute_bound_with_padding(flatten_ragged_float_column(table.column("atar_x"))),
        "y": compute_bound_with_padding(flatten_ragged_float_column(table.column("atar_y"))),
        "z": compute_bound_with_padding(flatten_ragged_float_column(table.column("atar_z"))),
    }


def compute_positron_zoom_bounds(truth, detector_bounds, axis_pair):
    beam_axis = axis_pair[0]
    transverse_axis = axis_pair[1]
    start = truth["positron_start"]
    direction = truth["direction"]
    if not (finite_triplet(start) and np.all(np.isfinite(direction))):
        return {
            beam_axis: detector_bounds[beam_axis],
            transverse_axis: detector_bounds[transverse_axis],
        }

    beam_idx = AXIS_INDEX[beam_axis]
    transverse_idx = AXIS_INDEX[transverse_axis]
    beam_lo, beam_hi = detector_bounds[beam_axis]
    trans_lo, trans_hi = detector_bounds[transverse_axis]
    beam_span = beam_hi - beam_lo
    trans_span = trans_hi - trans_lo

    beam_sign = 1.0 if direction[beam_idx] >= 0 else -1.0
    beam_forward = max(1.4, 0.34 * beam_span)
    beam_backward = max(0.8, 0.12 * beam_span)
    trans_half = max(1.2, 0.20 * trans_span)

    if beam_sign >= 0:
        beam_interval = (start[beam_idx] - beam_backward, start[beam_idx] + beam_forward)
    else:
        beam_interval = (start[beam_idx] - beam_forward, start[beam_idx] + beam_backward)

    trans_center = start[transverse_idx] + 0.10 * trans_span * np.clip(direction[transverse_idx], -1.0, 1.0)
    trans_interval = (trans_center - trans_half, trans_center + trans_half)

    return {
        beam_axis: clamp_interval(beam_interval, detector_bounds[beam_axis]),
        transverse_axis: clamp_interval(trans_interval, detector_bounds[transverse_axis]),
    }


def add_truth_direction(ax, truth, axis_pair, span_fraction=0.14):
    if not finite_triplet(truth["positron_start"]):
        return
    direction = truth["direction"]
    if not np.all(np.isfinite(direction)):
        return
    start = truth["positron_start"]
    i0 = AXIS_INDEX[axis_pair[0]]
    i1 = AXIS_INDEX[axis_pair[1]]
    proj = np.array([direction[i0], direction[i1]], dtype=float)
    proj_norm = float(np.linalg.norm(proj))
    if proj_norm <= 0.0 or not np.isfinite(proj_norm):
        return
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xspan = abs(x1 - x0)
    yspan = abs(y1 - y0)
    delta_x = span_fraction * xspan * (proj[0] / proj_norm)
    delta_y = span_fraction * yspan * (proj[1] / proj_norm)
    ax.annotate(
        "",
        xy=(start[i0] + delta_x, start[i1] + delta_y),
        xytext=(start[i0], start[i1]),
        arrowprops={
            "arrowstyle": "->",
            "color": "black",
            "lw": 1.8,
            "alpha": 0.9,
        },
    )
    scatter_truth_point(
        ax,
        start,
        axis_pair,
        color="black",
        label="positron direction start",
        marker="o",
        size=28,
        alpha=0.9,
        zorder=6,
    )


def add_truth_stop_stars(ax, truth, axis_pair):
    for name, color in [("pion_stop", "tab:green"), ("muon_stop", "tab:red")]:
        point = truth[name]
        if not finite_triplet(point):
            continue
        i0 = AXIS_INDEX[axis_pair[0]]
        i1 = AXIS_INDEX[axis_pair[1]]
        ax.scatter(
            [point[i0]],
            [point[i1]],
            s=260,
            marker="*",
            color=color,
            alpha=0.5,
            edgecolors="black",
            linewidths=0.6,
            label=f"{name} star",
            zorder=6,
        )


def add_truth_connectors(ax, truth, axis_pair, include_upstream):
    connector_specs = [
        ("pion_start", "pion_stop", "tab:green", "pion connector"),
        ("muon_start", "muon_stop", "tab:red", "muon connector"),
        ("positron_start", "positron_stop", "tab:purple", "positron connector"),
    ]
    i0 = AXIS_INDEX[axis_pair[0]]
    i1 = AXIS_INDEX[axis_pair[1]]
    for start_name, stop_name, color, label in connector_specs:
        start = truth[start_name]
        stop = truth[stop_name]
        if not (finite_triplet(start) and finite_triplet(stop)):
            continue
        ax.plot(
            [start[i0], stop[i0]],
            [start[i1], stop[i1]],
            color=color,
            linewidth=1.4,
            alpha=0.22,
            zorder=2,
            label=label,
        )


def add_truth_endpoints(ax, truth, axis_pair, include_upstream, emphasize_stops=False):
    endpoint_specs = [
        ("pion_start", "tab:green", "pion start", "^", 58, 0.65),
        ("pion_stop", "tab:green", "pion stop", "o", 58, 0.85),
        ("muon_start", "tab:red", "muon start", "^", 58, 0.65),
        ("muon_stop", "tab:red", "muon stop", "o", 58, 0.85),
        ("positron_start", "tab:purple", "positron start", "^", 62, 0.75),
        ("positron_stop", "tab:purple", "positron stop", "o", 62, 0.85),
    ]
    for name, color, label, marker, size, alpha in endpoint_specs:
        if name == "pion_start" and not include_upstream:
            continue
        scatter_truth_point(
            ax,
            truth[name],
            axis_pair,
            color=color,
            label=label,
            marker=marker,
            size=size,
            alpha=alpha,
        )
    if emphasize_stops:
        add_truth_stop_stars(ax, truth, axis_pair)


def finalize_axis(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        dedup = {}
        for handle, label in zip(handles, labels):
            if label not in dedup:
                dedup[label] = handle
        ax.legend(
            list(dedup.values()),
            list(dedup.keys()),
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            fontsize=8,
            frameon=False,
        )


def render_spatial_panel(
    ax,
    atar,
    lyso,
    truth,
    axis_pair,
    title,
    include_upstream,
    local,
    zoom_margin,
    show_lyso=True,
    fixed_limits=None,
    equal_aspect=True,
    emphasize_stops=False,
):
    beam_axis = axis_pair[0]
    transverse_axis = axis_pair[1]
    x_mask = (atar["view"] == 0) if len(atar["view"]) else np.zeros(len(atar["x"]), dtype=bool)
    y_mask = (atar["view"] == 1) if len(atar["view"]) else np.zeros(len(atar["x"]), dtype=bool)

    if transverse_axis == "x" and np.any(x_mask):
        scatter_hits_by_pdg(
            ax,
            atar[beam_axis][x_mask],
            atar[transverse_axis][x_mask],
            atar["pdg"][x_mask],
            base_label="ATAR x-view",
            size=18,
            alpha=0.85,
        )
    if transverse_axis == "y" and np.any(y_mask):
        scatter_hits_by_pdg(
            ax,
            atar[beam_axis][y_mask],
            atar[transverse_axis][y_mask],
            atar["pdg"][y_mask],
            base_label="ATAR y-view",
            size=18,
            alpha=0.85,
        )

    if show_lyso and len(lyso["x"]) > 0:
        sizes = np.clip(lyso["e"] * 3.0, 15.0, 80.0)
        scatter_hits_by_pdg(
            ax,
            lyso[beam_axis],
            lyso[transverse_axis],
            lyso["pdg"],
            base_label="LYSO",
            marker="x",
            alpha=0.45,
            size_scale=sizes,
        )

    if fixed_limits is None:
        points = collect_xyz_points(atar, lyso, truth, include_upstream=include_upstream, local=local)
        xlim, ylim = compute_limits(points, axis_pair, margin_fraction=zoom_margin)
    else:
        xlim = fixed_limits[axis_pair[0]]
        ylim = fixed_limits[axis_pair[1]]
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal" if equal_aspect else "auto", adjustable="box")

    add_truth_connectors(
        ax,
        truth,
        axis_pair,
        include_upstream=include_upstream or not local,
    )
    add_truth_endpoints(
        ax,
        truth,
        axis_pair,
        include_upstream=include_upstream or not local,
        emphasize_stops=emphasize_stops,
    )
    add_truth_direction(ax, truth, axis_pair, span_fraction=0.12 if local else 0.14)

    ax.set_title(title)
    ax.set_xlabel(f"{axis_pair[0]} [mm]")
    ax.set_ylabel(f"{axis_pair[1]} [mm]")
    finalize_axis(ax)


def render_timing_panel(ax, atar, lyso, truth):
    if len(atar["t"]) > 0:
        scatter_hits_by_pdg(
            ax,
            atar["z"],
            atar["t"],
            atar["pdg"],
            base_label="ATAR observed t",
            size=18,
            alpha=0.8,
        )
    if len(atar["truth_t"]) == len(atar["t"]) and len(atar["t"]) > 0:
        scatter_hits_by_pdg(
            ax,
            atar["z"],
            atar["truth_t"],
            atar["pdg"],
            base_label="ATAR truth t",
            size=10,
            alpha=0.35,
            marker="x",
        )
    if len(lyso["t"]) > 0:
        sizes = np.clip(lyso["e"] * 3.0, 15.0, 80.0)
        scatter_hits_by_pdg(
            ax,
            lyso["z"],
            lyso["t"],
            lyso["pdg"],
            base_label="LYSO",
            marker="x",
            alpha=0.5,
            size_scale=sizes,
        )

    for name, color in [
        ("pion_stop", "tab:green"),
        ("muon_stop", "tab:red"),
        ("positron_start", "tab:purple"),
        ("positron_stop", "tab:purple"),
    ]:
        point = truth[name]
        if finite_triplet(point):
            ax.axvline(point[2], color=color, linestyle="--", linewidth=1.1, alpha=0.6)

    ax.set_title("Detector timing vs z")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("t [ns]")
    finalize_axis(ax)


def render_dt_panel(ax, atar):
    has_truth_t = len(atar["truth_t"]) == len(atar["t"]) and len(atar["t"]) > 0
    if has_truth_t:
        dt = atar["t"] - atar["truth_t"]
        ax.hist(dt, bins=min(50, max(12, len(dt) // 2)), color="tab:blue", alpha=0.8)
        ax.axvline(np.mean(dt), color="black", linestyle="--", linewidth=1.2, label=f"mean={np.mean(dt):.4f} ns")
        ax.set_title("ATAR observed - truth time")
        ax.set_xlabel("t_obs - t_truth [ns]")
        ax.set_ylabel("count")
        finalize_axis(ax)
        return

    ax.text(0.5, 0.5, "No ATAR truth timing column available", ha="center", va="center")
    ax.set_axis_off()


def make_figure(figsize):
    return plt.subplots(figsize=figsize, constrained_layout=True)


def render_spatial_figure(
    event,
    axis_pair,
    title,
    include_upstream,
    local,
    zoom_margin,
    show_lyso=True,
    fixed_limits=None,
    equal_aspect=True,
    emphasize_stops=False,
):
    fig, ax = make_figure(SPATIAL_FIGSIZE)
    render_spatial_panel(
        ax,
        event.atar,
        event.lyso,
        event.truth,
        axis_pair,
        title,
        include_upstream=include_upstream,
        local=local,
        zoom_margin=zoom_margin,
        show_lyso=show_lyso,
        fixed_limits=fixed_limits,
        equal_aspect=equal_aspect,
        emphasize_stops=emphasize_stops,
    )
    return fig


def render_timing_figure(event):
    fig, ax = make_figure(TIME_FIGSIZE)
    render_timing_panel(ax, event.atar, event.lyso, event.truth)
    return fig


def render_dt_figure(event):
    fig, ax = make_figure(TIME_FIGSIZE)
    render_dt_panel(ax, event.atar)
    return fig


def render_event_report(
    parquet_path=DEFAULT_PARQUET_PATH,
    row_index=DEFAULT_ROW_INDEX,
    include_upstream_atar_section=False,
):
    path_str = str(Path(parquet_path))
    event = load_event_data(path_str, int(row_index))
    atar_section_bounds = load_atar_section_bounds(path_str)
    positron_zoom_zx = compute_positron_zoom_bounds(event.truth, atar_section_bounds, "zx")
    positron_zoom_zy = compute_positron_zoom_bounds(event.truth, atar_section_bounds, "zy")
    display(HTML(build_summary_html(event, atar_section_bounds=atar_section_bounds)))

    figures = [
        render_spatial_figure(
            event,
            axis_pair="zx",
            title="Global ZX view",
            include_upstream=True,
            local=False,
            zoom_margin=0.12,
        ),
        render_spatial_figure(
            event,
            axis_pair="zy",
            title="Global ZY view",
            include_upstream=True,
            local=False,
            zoom_margin=0.12,
        ),
        render_timing_figure(event),
        render_spatial_figure(
            event,
            axis_pair="zx",
            title="ATAR section ZX",
            include_upstream=include_upstream_atar_section,
            local=True,
            zoom_margin=0.0,
            show_lyso=False,
            fixed_limits=atar_section_bounds,
            equal_aspect=False,
            emphasize_stops=True,
        ),
        render_spatial_figure(
            event,
            axis_pair="zy",
            title="ATAR section ZY",
            include_upstream=include_upstream_atar_section,
            local=True,
            zoom_margin=0.0,
            show_lyso=False,
            fixed_limits=atar_section_bounds,
            equal_aspect=False,
            emphasize_stops=True,
        ),
        render_spatial_figure(
            event,
            axis_pair="zx",
            title="Positron-angle zoom ZX",
            include_upstream=include_upstream_atar_section,
            local=True,
            zoom_margin=0.0,
            show_lyso=False,
            fixed_limits=positron_zoom_zx,
            equal_aspect=False,
            emphasize_stops=True,
        ),
        render_spatial_figure(
            event,
            axis_pair="zy",
            title="Positron-angle zoom ZY",
            include_upstream=include_upstream_atar_section,
            local=True,
            zoom_margin=0.0,
            show_lyso=False,
            fixed_limits=positron_zoom_zy,
            equal_aspect=False,
            emphasize_stops=True,
        ),
        render_dt_figure(event),
    ]

    for fig in figures:
        display(fig)
        plt.close(fig)

    return event


def format_point(point):
    if not finite_triplet(point):
        return "n/a"
    return f"[{point[0]:8.3f}, {point[1]:8.3f}, {point[2]:8.3f}]"


def format_float(value):
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def build_summary_text(event, atar_section_bounds=None):
    row = event.row
    truth = event.truth
    atar = event.atar
    lyso = event.lyso

    lines = [
        f"Row {event.row_index}",
        f"event_id={optional_col(row, 'event_id', 'UNKNOWN')}",
        f"schema_version={optional_col(row, 'schema_version', 'UNKNOWN')}",
        f"ATAR hits={len(atar['x'])}, LYSO hits={len(lyso['x'])}",
        (
            "truth_theta="
            f"{format_float(truth['theta'])}, truth_phi={format_float(truth['phi'])}, "
            f"truth_positron_energy={format_float(truth['positron_energy'])}"
        ),
        "",
        "Truth vertices:",
    ]
    for name in ["pion_start", "pion_stop", "muon_start", "muon_stop", "positron_start", "positron_stop"]:
        lines.append(f"  {name:16s} = {format_point(truth[name])}")
    lines.extend(
        [
            "",
            "Segment lengths [mm]:",
            f"  pion      = {format_float(segment_length(truth['pion_start'], truth['pion_stop']))}",
            f"  muon      = {format_float(segment_length(truth['muon_start'], truth['muon_stop']))}",
            f"  positron  = {format_float(segment_length(truth['positron_start'], truth['positron_stop']))}",
            "",
        ]
    )
    if atar_section_bounds is not None:
        lines.extend(
            [
                "ATAR section bounds [mm]:",
                f"  z = [{atar_section_bounds['z'][0]:.3f}, {atar_section_bounds['z'][1]:.3f}]",
                f"  x = [{atar_section_bounds['x'][0]:.3f}, {atar_section_bounds['x'][1]:.3f}]",
                f"  y = [{atar_section_bounds['y'][0]:.3f}, {atar_section_bounds['y'][1]:.3f}]",
                "",
                "ATAR-section plots use these fixed detector bounds and hide LYSO hits.",
            ]
        )
    return "\n".join(lines)


def build_summary_html(event, atar_section_bounds=None):
    summary_text = build_summary_text(event, atar_section_bounds=atar_section_bounds)
    return (
        "<div style='border:1px solid #d8d8d8; padding:12px; background:#fcfcfc;'>"
        "<pre style='margin:0; white-space:pre-wrap; font-family:monospace;'>"
        f"{escape(summary_text)}"
        "</pre>"
        "</div>"
    )


@lru_cache(maxsize=8)
def load_file_index(path_str):
    parquet_table = load_table(path_str)
    num_rows = parquet_table.num_rows
    try:
        event_ids = tuple(parquet_table.column("event_id").to_pylist())
    except Exception:
        event_ids = tuple(range(num_rows))
    return num_rows, event_ids


@lru_cache(maxsize=128)
def load_event_data(path_str, row_index):
    row = load_row_dict(Path(path_str), row_index)
    return EventData(
        row_index=row_index,
        row=row,
        truth=build_truth_payload(row),
        atar=extract_detector_payload(row, "atar"),
        lyso=extract_detector_payload(row, "lyso"),
    )


def make_output_widget():
    return widgets.Output(
        layout=widgets.Layout(
            width="100%",
            min_height="420px",
            overflow="hidden",
        )
    )


def make_plot_card(title, output_widget):
    header = widgets.HTML(f"<div style='font-weight:600; margin-bottom:8px;'>{escape(title)}</div>")
    return widgets.VBox(
        [header, output_widget],
        layout=widgets.Layout(
            width="100%",
            border="1px solid #d8d8d8",
            padding="12px",
            align_items="stretch",
        ),
    )


class TruthDetectorValidationDashboard:
    def __init__(self, parquet_path=DEFAULT_PARQUET_PATH, default_row_index=DEFAULT_ROW_INDEX):
        self.parquet_path = Path(parquet_path)
        self.default_row_index = int(default_row_index)

        self.row_info = widgets.HTML()
        self.summary = widgets.HTML()
        self.status = widgets.HTML()

        if self.parquet_path.exists():
            self.num_rows, self.event_ids = load_file_index(str(self.parquet_path))
            self.atar_section_bounds = load_atar_section_bounds(str(self.parquet_path))
        else:
            self.num_rows, self.event_ids = 0, tuple()
            self.atar_section_bounds = None

        max_index = max(0, self.num_rows - 1)
        initial_row = min(max(self.default_row_index, 0), max_index)

        self.row_slider = widgets.IntSlider(
            value=initial_row,
            min=0,
            max=max_index,
            step=1,
            description="Row",
            continuous_update=False,
            layout=widgets.Layout(width="100%"),
        )
        self.row_text = widgets.BoundedIntText(
            value=initial_row,
            min=0,
            max=max_index,
            step=1,
            description="Row #",
            layout=widgets.Layout(width="180px"),
        )
        widgets.link((self.row_slider, "value"), (self.row_text, "value"))

        self.include_upstream_checkbox = widgets.Checkbox(
            value=False,
            description="Include upstream pion start in ATAR-section plots",
        )

        self.global_xz_output = make_output_widget()
        self.global_yz_output = make_output_widget()
        self.timing_output = make_output_widget()
        self.local_xz_output = make_output_widget()
        self.local_yz_output = make_output_widget()
        self.positron_zoom_xz_output = make_output_widget()
        self.positron_zoom_yz_output = make_output_widget()
        self.dt_output = make_output_widget()

        self.widget = self._build_layout()
        self._bind()
        self.refresh()

    def _build_layout(self):
        controls = widgets.VBox(
            [
                self.row_info,
                widgets.HBox([self.row_slider, self.row_text], layout=widgets.Layout(width="100%")),
                self.include_upstream_checkbox,
                self.status,
            ],
            layout=widgets.Layout(gap="10px"),
        )

        plot_grid = widgets.GridBox(
            children=[
                make_plot_card("Global ZX view", self.global_xz_output),
                make_plot_card("Global ZY view", self.global_yz_output),
                make_plot_card("Detector timing vs z", self.timing_output),
                make_plot_card("ATAR section ZX", self.local_xz_output),
                make_plot_card("ATAR section ZY", self.local_yz_output),
                make_plot_card("Positron-angle zoom ZX", self.positron_zoom_xz_output),
                make_plot_card("Positron-angle zoom ZY", self.positron_zoom_yz_output),
                make_plot_card("ATAR observed - truth time", self.dt_output),
            ],
            layout=widgets.Layout(
                width="100%",
                grid_template_columns="repeat(auto-fit, minmax(460px, 1fr))",
                gap="16px",
                align_items="flex-start",
            ),
        )

        return widgets.VBox(
            [controls, self.summary, plot_grid],
            layout=widgets.Layout(width="100%", gap="16px"),
        )

    def _bind(self):
        for control in [self.row_slider, self.include_upstream_checkbox]:
            control.observe(self.refresh, names="value")

    def _event_id_for_row(self, row_index):
        if row_index < len(self.event_ids):
            return self.event_ids[row_index]
        return "n/a"

    def _update_row_info(self):
        row_index = self.row_slider.value
        self.row_info.value = (
            f"<b>File:</b> {escape(self.parquet_path.name)} &nbsp; "
            f"<b>row:</b> {row_index} &nbsp; "
            f"<b>event_id:</b> {escape(str(self._event_id_for_row(row_index)))} &nbsp; "
            f"<b>rows in file:</b> {self.num_rows}"
        )

    def _render_plot(self, output_widget, figure_factory):
        output_widget.clear_output(wait=True)
        fig = figure_factory()
        with output_widget:
            display(fig)
        plt.close(fig)

    def _clear_plot_outputs(self):
        for output_widget in [
            self.global_xz_output,
            self.global_yz_output,
            self.timing_output,
            self.local_xz_output,
            self.local_yz_output,
            self.positron_zoom_xz_output,
            self.positron_zoom_yz_output,
            self.dt_output,
        ]:
            output_widget.clear_output(wait=True)

    def refresh(self, *_):
        self._update_row_info()

        if not self.parquet_path.exists():
            self.status.value = (
                "<div style='color:#b00020;'><b>Missing parquet file:</b> "
                f"{escape(str(self.parquet_path))}</div>"
            )
            self.summary.value = ""
            self._clear_plot_outputs()
            return

        try:
            event = load_event_data(str(self.parquet_path), self.row_slider.value)
        except Exception as exc:
            self.status.value = (
                "<div style='color:#b00020;'><b>Render failed:</b> "
                f"{escape(str(exc))}</div>"
            )
            self.summary.value = ""
            self._clear_plot_outputs()
            return

        self.status.value = ""
        self.summary.value = build_summary_html(event, atar_section_bounds=self.atar_section_bounds)
        positron_zoom_zx = compute_positron_zoom_bounds(event.truth, self.atar_section_bounds, "zx")
        positron_zoom_zy = compute_positron_zoom_bounds(event.truth, self.atar_section_bounds, "zy")
        self._render_plot(
            self.global_xz_output,
            lambda: render_spatial_figure(
                event,
                axis_pair="zx",
                title="Global ZX view",
                include_upstream=True,
                local=False,
                zoom_margin=0.12,
            ),
        )
        self._render_plot(
            self.global_yz_output,
            lambda: render_spatial_figure(
                event,
                axis_pair="zy",
                title="Global ZY view",
                include_upstream=True,
                local=False,
                zoom_margin=0.12,
            ),
        )
        self._render_plot(self.timing_output, lambda: render_timing_figure(event))
        self._render_plot(
            self.local_xz_output,
            lambda: render_spatial_figure(
                event,
                axis_pair="zx",
                title="ATAR section ZX",
                include_upstream=self.include_upstream_checkbox.value,
                local=True,
                zoom_margin=0.0,
                show_lyso=False,
                fixed_limits=self.atar_section_bounds,
                equal_aspect=False,
                emphasize_stops=True,
            ),
        )
        self._render_plot(
            self.local_yz_output,
            lambda: render_spatial_figure(
                event,
                axis_pair="zy",
                title="ATAR section ZY",
                include_upstream=self.include_upstream_checkbox.value,
                local=True,
                zoom_margin=0.0,
                show_lyso=False,
                fixed_limits=self.atar_section_bounds,
                equal_aspect=False,
                emphasize_stops=True,
            ),
        )
        self._render_plot(
            self.positron_zoom_xz_output,
            lambda: render_spatial_figure(
                event,
                axis_pair="zx",
                title="Positron-angle zoom ZX",
                include_upstream=self.include_upstream_checkbox.value,
                local=True,
                zoom_margin=0.0,
                show_lyso=False,
                fixed_limits=positron_zoom_zx,
                equal_aspect=False,
                emphasize_stops=True,
            ),
        )
        self._render_plot(
            self.positron_zoom_yz_output,
            lambda: render_spatial_figure(
                event,
                axis_pair="zy",
                title="Positron-angle zoom ZY",
                include_upstream=self.include_upstream_checkbox.value,
                local=True,
                zoom_margin=0.0,
                show_lyso=False,
                fixed_limits=positron_zoom_zy,
                equal_aspect=False,
                emphasize_stops=True,
            ),
        )
        self._render_plot(self.dt_output, lambda: render_dt_figure(event))


def create_validation_dashboard(parquet_path=DEFAULT_PARQUET_PATH, default_row_index=DEFAULT_ROW_INDEX):
    return TruthDetectorValidationDashboard(parquet_path=parquet_path, default_row_index=default_row_index)
