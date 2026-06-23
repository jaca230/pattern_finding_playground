from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from figures.event_display.event_display_config import EventDisplayConfig
from figures.event_display.event_display_data import HitGlyph
from figures.event_display.geometry import AtarGeometryProjector
from figures.event_display.renderers import (
    HitRenderer,
    LegendRenderer,
    TrackletRenderer,
    VertexRenderer,
)
from figures.event_display.timing import TimeClusterer
from figures.figure import PlotFigure
from models.event import Event


class EventDisplayFigure(PlotFigure):
    def __init__(self, config: EventDisplayConfig | None = None, show: bool = True):
        super().__init__(show=show)
        self.config = config or EventDisplayConfig()
        self.hit_renderer = HitRenderer()
        self.tracklet_renderer = TrackletRenderer()
        self.vertex_renderer = VertexRenderer()
        self.legend_renderer = LegendRenderer()

    def draw(self, event: Event):
        tracklets = self._tracklets_for_event(event)
        vertices = self._vertices_for_event(event)
        all_hits = list(event.all_hits) if getattr(event, "all_hits", None) else [hit for tracklet in tracklets for hit in tracklet.hits]

        projector = AtarGeometryProjector(event.extra_info.get("geo"))
        xz_glyphs = self._glyphs_for_plane(all_hits, projector, "xz")
        yz_glyphs = self._glyphs_for_plane(all_hits, projector, "yz")

        extra_rows = int(self.config.show_time_panel) + int(self.config.show_energy_panel)
        n_rows = 2 + extra_rows
        height_ratios = [1.0, 1.0]
        if self.config.show_time_panel:
            height_ratios.append(0.78)
        if self.config.show_energy_panel:
            height_ratios.append(0.78)
        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=self.config.figsize,
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": height_ratios, "hspace": 0.13},
        )
        axes = np.atleast_1d(axes)
        z_limits = self._shared_z_limits(xz_glyphs, yz_glyphs, tracklets, vertices, all_hits)

        self._draw_spatial_panel(
            axes[0],
            xz_glyphs,
            tracklets,
            vertices,
            projector,
            "xz",
            "x [mm]",
            z_limits,
        )
        self._draw_spatial_panel(
            axes[1],
            yz_glyphs,
            tracklets,
            vertices,
            projector,
            "yz",
            "y [mm]",
            z_limits,
        )

        row = 2
        if self.config.show_time_panel:
            self._draw_time_panel(axes[row], all_hits, z_limits)
            row += 1
        if self.config.show_energy_panel:
            self._draw_energy_panel(axes[row], all_hits, z_limits)

        axes[-1].set_xlabel("z [mm]")
        self.legend_renderer.draw_particle_legend(
            axes[0],
            [hit.particle_id for hit in all_hits],
            include_vertex=True,
            include_endpoint=True,
            title="hits",
        )

        self._display(plt)
        return fig, axes

    def _draw_spatial_panel(
        self,
        ax,
        glyphs: list[HitGlyph],
        tracklets,
        vertices,
        projector: AtarGeometryProjector,
        plane: str,
        coord_label: str,
        z_limits: tuple[float, float],
    ) -> None:
        ax.set_ylabel(coord_label)
        self._set_spatial_limits(ax, glyphs, tracklets, vertices, plane, z_limits)
        self.hit_renderer.draw_layer_guides(ax, projector.guide_rectangles(glyphs, plane), self.config)
        self.hit_renderer.draw_particle_spatial_hits(ax, glyphs, self.config)
        self.tracklet_renderer.draw(ax, tracklets, plane, self.config)
        self.vertex_renderer.draw(ax, vertices, plane, self.config)
        ax.grid(alpha=0.18)

    def _draw_time_panel(self, ax, hits, z_limits: tuple[float, float]) -> None:
        ax.set_ylabel("time [ns]")
        time_clusterer = TimeClusterer(
            gap_threshold_ns=self.config.time_gap_threshold_ns,
            bridge_threshold_ns=self.config.time_bridge_threshold_ns,
            gap_fraction=self.config.time_gap_fraction,
            min_display_gap_ns=self.config.time_gap_min_display_ns,
        )
        clusters = time_clusterer.cluster(hits)
        self.hit_renderer.draw_time_hits(ax, hits, time_clusterer, clusters, self.config)
        self._draw_time_cluster_breaks(ax, clusters, z_limits)
        self._set_time_limits(ax, clusters, z_limits)
        self.legend_renderer.draw_particle_legend(
            ax,
            [hit.particle_id for hit in hits],
            include_vertex=False,
            title="hits",
        )
        ax.grid(alpha=0.18)

    def _draw_energy_panel(self, ax, hits, z_limits: tuple[float, float]) -> None:
        ax.set_ylabel("energy [MeV]")
        self.hit_renderer.draw_energy_hits(ax, hits, self.config)
        ax.set_yscale("log")
        ax.set_xlim(z_limits)
        self.legend_renderer.draw_particle_legend(
            ax,
            [hit.particle_id for hit in hits],
            include_vertex=False,
            title="hits",
        )
        ax.grid(alpha=0.18)

    def _draw_time_cluster_breaks(self, ax, clusters, z_limits: tuple[float, float]) -> None:
        if not clusters:
            return
        tick_positions = []
        tick_labels = []
        for cluster in clusters:
            tick_positions.extend([cluster.display_start, cluster.display_stop])
            tick_labels.extend([f"{cluster.raw_start:.2f}", f"{cluster.raw_stop:.2f}"])

        for left_cluster, right_cluster in zip(clusters, clusters[1:]):
            gap_start = left_cluster.display_stop
            gap_stop = right_cluster.display_start
            if gap_stop <= gap_start:
                continue
            ax.fill_between(
                z_limits,
                gap_start,
                gap_stop,
                color="0.86",
                alpha=0.55,
                hatch="//",
                linewidth=0,
                zorder=1,
            )

        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

    def _glyphs_for_plane(self, hits, projector: AtarGeometryProjector, plane: str) -> list[HitGlyph]:
        expected_side = "front" if plane == "xz" else "back"
        glyphs = []
        for hit in hits:
            if hit.detector_side != expected_side:
                continue
            glyphs.append(
                HitGlyph(
                    hit=hit,
                    tracklet=None,
                    plane=plane,
                    rectangle=projector.rectangle_for_hit(hit, plane),
                    energy=float(hit.energy or self.config.energy_floor_mev),
                    time=hit.time,
                    particle_id=hit.particle_id,
                )
            )
        return glyphs

    def _tracklets_for_event(self, event: Event):
        if event.get_patterns():
            return {tracklet for pattern in event.get_patterns() for tracklet in pattern.get_unique_tracklets()}
        return event.all_tracklets

    def _vertices_for_event(self, event: Event):
        if event.get_patterns():
            vertices = {
                vertex
                for pattern in event.get_patterns()
                for vertex in pattern.get_vertices()
            }
            if vertices:
                return vertices
        return getattr(event, "all_vertices", set())

    def _set_spatial_limits(self, ax, glyphs, tracklets, vertices, plane: str, z_limits: tuple[float, float]) -> None:
        coord_values = []
        for glyph in glyphs:
            coord_values.extend([glyph.rectangle.coord_min, glyph.rectangle.coord_max])

        for tracklet in tracklets:
            for _, coord in self._display_endpoints_for_tracklet(tracklet, plane):
                if self._finite_display_value(coord):
                    coord_values.append(coord)

        position_key = "front_vertex_position" if plane == "xz" else "back_vertex_position"
        for vertex in vertices:
            position = vertex.extra_info.get(position_key)
            if position is None:
                continue
            z = position[2]
            coord = position[0] if plane == "xz" else position[1]
            if self._finite_display_value(z) and self._finite_display_value(coord):
                coord_values.append(coord)

        ax.set_xlim(z_limits)
        ax.set_ylim(self._padded_limits(coord_values, default=(-8.0, 8.0)))

    def _set_time_limits(self, ax, clusters, z_limits: tuple[float, float]) -> None:
        ax.set_xlim(z_limits)
        if clusters:
            y0 = min(cluster.display_start for cluster in clusters)
            y1 = max(cluster.display_stop for cluster in clusters)
            ax.set_ylim(self._padded_limits([y0, y1], default=(0.0, 1.0)))

    def _shared_z_limits(self, xz_glyphs, yz_glyphs, tracklets, vertices, hits) -> tuple[float, float]:
        z_values = []
        for glyph in xz_glyphs + yz_glyphs:
            z_values.extend([glyph.rectangle.z_min, glyph.rectangle.z_max])

        for hit in hits:
            if self._finite_display_value(hit.z):
                z_values.append(hit.z)

        for tracklet in tracklets:
            for z, _ in self._display_endpoints_for_tracklet(tracklet, "xz"):
                if self._finite_display_value(z):
                    z_values.append(z)

        for vertex in vertices:
            for key in ("front_vertex_position", "back_vertex_position"):
                position = vertex.extra_info.get(key)
                if position is not None and self._finite_display_value(position[2]):
                    z_values.append(position[2])

        return self._padded_limits(z_values, default=(0.0, 7.0))

    def _padded_limits(self, values, default: tuple[float, float]) -> tuple[float, float]:
        finite = [float(value) for value in values if self._finite_display_value(value)]
        if not finite:
            return default
        low = min(finite)
        high = max(finite)
        if low == high:
            return low - self.config.minimum_spatial_margin, high + self.config.minimum_spatial_margin
        margin = max((high - low) * self.config.spatial_margin_fraction, self.config.minimum_spatial_margin)
        return low - margin, high + margin

    def _finite_display_value(self, value) -> bool:
        return value is not None and math.isfinite(float(value)) and abs(float(value)) < 1.0e6

    def _display_endpoints_for_tracklet(self, tracklet, plane: str) -> list[tuple[float, float]]:
        display_endpoints = tracklet.extra_info.get("display_endpoints")
        if isinstance(display_endpoints, dict):
            return [
                point
                for point in display_endpoints.get(plane, [])
                if self._finite_display_value(point[0]) and self._finite_display_value(point[1])
            ]
        return []
