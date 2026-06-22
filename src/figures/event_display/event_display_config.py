from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EventDisplayConfig:
    figsize: tuple[float, float] = (13.0, 10.0)
    hit_alpha: float = 0.72
    hit_edge_width: float = 1.05
    hit_edge_color: str = "#555555"
    tracklet_line_width: float = 3.0
    tracklet_line_alpha: float = 0.82
    tracklet_label_fontsize: float = 11.0
    endpoint_marker_size: float = 74.0
    endpoint_label_fontsize: float = 8.0
    endpoint_z_uncertainty_mm: float = 0.035
    endpoint_coord_uncertainty_mm: float = 0.035
    vertex_marker_size: float = 95.0
    strip_count_fontsize: float = 8.0
    geometry_guide_alpha: float = 0.14
    energy_floor_mev: float = 1.0e-4
    energy_cmap: str = "viridis"
    time_gap_threshold_ns: float = 0.3
    time_bridge_threshold_ns: float = 0.95
    time_gap_fraction: float = 0.18
    time_gap_min_display_ns: float = 0.32
    show_time_panel: bool = True
    show_energy_panel: bool = True
    show_tracklets: bool = True
    show_tracklet_labels: bool = False
    show_endpoint_labels: bool = True
    show_start_endpoint_labels: bool = False
    show_vertices: bool = True
    show_patterns: bool = False
    spatial_margin_fraction: float = 0.08
    minimum_spatial_margin: float = 0.25
