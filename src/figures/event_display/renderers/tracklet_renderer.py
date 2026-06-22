from __future__ import annotations

import math

import numpy as np

from figures.event_display.styles import particle_style, tracklet_line_color


class TrackletRenderer:
    def draw(self, ax, tracklets, plane: str, config) -> None:
        if not config.show_tracklets:
            return

        for tracklet in tracklets:
            line_points, endpoint_points, has_reconstructed_endpoints = self._display_points_for_tracklet(
                tracklet,
                plane,
            )
            if len(line_points) < 2:
                continue
            if not all(self._is_valid_point(point) for point in line_points):
                continue

            color = tracklet_line_color(tracklet.particle_id)
            ax.plot(
                [point[0] for point in line_points],
                [point[1] for point in line_points],
                color=color,
                linewidth=config.tracklet_line_width,
                alpha=config.tracklet_line_alpha,
                zorder=8,
            )
            if has_reconstructed_endpoints:
                self._draw_endpoint_markers(ax, endpoint_points, color, config)
            if config.show_tracklet_labels:
                self._draw_label(ax, line_points, particle_style(tracklet.particle_id)["name"], color, config)

    def _display_points_for_tracklet(
        self,
        tracklet,
        plane: str,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]], bool]:
        hit_points = self._hit_points_for_tracklet(tracklet, plane)
        line_points = self._straight_line_through_hits(hit_points)

        ep0, ep1 = tracklet.get_endpoints()
        if ep0 and ep1:
            endpoint_points = [self._point_for_plane(ep0, plane), self._point_for_plane(ep1, plane)]
            if len(line_points) >= 2:
                return line_points, endpoint_points, True
            return endpoint_points, endpoint_points, True

        return line_points, [], False

    def _hit_points_for_tracklet(self, tracklet, plane: str) -> list[tuple[float, float]]:
        expected_side = "front" if plane == "xz" else "back"
        hit_points = []
        for hit in tracklet.hits:
            if hit.detector_side != expected_side:
                continue
            coord = hit.x if plane == "xz" else hit.y
            if coord is None or hit.z is None:
                continue
            hit_points.append((float(hit.z), float(coord)))
        return sorted(hit_points, key=lambda point: point[0])

    def _straight_line_through_hits(self, hit_points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        clean_points = [point for point in hit_points if self._is_valid_point(point)]
        if len(clean_points) < 2:
            return clean_points

        z_values = np.array([point[0] for point in clean_points], dtype=float)
        coord_values = np.array([point[1] for point in clean_points], dtype=float)
        z0 = float(np.min(z_values))
        z1 = float(np.max(z_values))
        if math.isclose(z0, z1):
            return [clean_points[0], clean_points[-1]]

        slope, intercept = np.polyfit(z_values, coord_values, deg=1)
        return [(z0, float(slope * z0 + intercept)), (z1, float(slope * z1 + intercept))]

    def _draw_endpoint_markers(self, ax, points, color: str, config) -> None:
        if len(points) < 2:
            return
        stop = points[-1]
        ax.errorbar(
            stop[0],
            stop[1],
            xerr=config.endpoint_z_uncertainty_mm,
            yerr=config.endpoint_coord_uncertainty_mm,
            fmt="+",
            color=color,
            ecolor=color,
            elinewidth=1.7,
            capsize=2.4,
            markersize=10.0,
            markeredgewidth=2.0,
            alpha=0.95,
            zorder=9,
        )
        if config.show_endpoint_labels:
            self._draw_endpoint_label(ax, stop, "end", color, config, vertical_alignment="bottom")

    def _draw_endpoint_label(
        self,
        ax,
        point: tuple[float, float],
        label: str,
        color: str,
        config,
        vertical_alignment: str,
    ) -> None:
        y_offset = -4 if vertical_alignment == "top" else 4
        ax.annotate(
            label,
            xy=point,
            xytext=(4, y_offset),
            textcoords="offset points",
            color=color,
            fontsize=config.endpoint_label_fontsize,
            ha="left",
            va=vertical_alignment,
            zorder=11,
            clip_on=True,
        )

    def _draw_label(self, ax, points, label: str, color: str, config) -> None:
        midpoint = points[len(points) // 2]
        ax.text(
            midpoint[0],
            midpoint[1],
            f"${label}$",
            color=color,
            fontsize=config.tracklet_label_fontsize,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": color, "alpha": 0.78},
            zorder=11,
            clip_on=True,
        )

    def _point_for_plane(self, point, plane: str) -> tuple[float, float]:
        if plane == "xz":
            return point.z, point.x
        if plane == "yz":
            return point.z, point.y
        raise ValueError(f"Unknown plane: {plane}")

    def _is_valid_point(self, point: tuple[float, float]) -> bool:
        return all(math.isfinite(value) and abs(value) < 1.0e6 for value in point)
