from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from figures.event_display.event_display_config import EventDisplayConfig
from figures.event_display.event_display_data import HitGlyph
from figures.event_display.geometry import AtarGeometryProjector
from figures.event_display.renderers import HitRenderer
from figures.event_display.renderers.legend_renderer import LegendRenderer
from figures.event_display.styles import EnergyStyle
from figures.event_display.timing import TimeClusterer
from figures.figure import PlotFigure
from models.event import Event


class DetectorEventViewFigure(PlotFigure):
    def __init__(self, config: EventDisplayConfig | None = None, show: bool = True):
        super().__init__(show=show)
        self.config = config or EventDisplayConfig()
        self.hit_renderer = HitRenderer()
        self.legend_renderer = LegendRenderer()

    def draw(self, event: Event):
        hits = list(event.all_hits)
        projector = AtarGeometryProjector(event.extra_info.get("geo"))
        xz_glyphs = self._glyphs_for_plane(hits, projector, "xz")
        yz_glyphs = self._glyphs_for_plane(hits, projector, "yz")
        energy_style = EnergyStyle(
            [glyph.energy for glyph in xz_glyphs + yz_glyphs],
            cmap=self.config.energy_cmap,
            floor=self.config.energy_floor_mev,
        )

        fig, axes = plt.subplots(
            4,
            1,
            figsize=self.config.figsize,
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1.0, 1.0, 0.78, 0.78], "hspace": 0.13},
        )
        axes = np.atleast_1d(axes)
        z_limits = self._shared_z_limits(xz_glyphs, yz_glyphs, hits)

        self._draw_spatial_panel(axes[0], xz_glyphs, projector, "xz", "x [mm]", energy_style, z_limits)
        self._draw_spatial_panel(axes[1], yz_glyphs, projector, "yz", "y [mm]", energy_style, z_limits)
        self._draw_time_panel(axes[2], hits, energy_style, z_limits)
        self._draw_energy_panel(axes[3], hits, energy_style, z_limits)

        axes[-1].set_xlabel("z [mm]")
        self.legend_renderer.draw_energy_colorbar(fig, axes[:2], energy_style)
        self._display(plt)
        return fig, axes

    def _draw_spatial_panel(self, ax, glyphs, projector, plane, coord_label, energy_style, z_limits):
        ax.set_ylabel(coord_label)
        self._set_spatial_limits(ax, glyphs, z_limits)
        self.hit_renderer.draw_layer_guides(ax, projector.guide_rectangles(glyphs, plane), self.config)
        self.hit_renderer.draw_energy_spatial_hits(ax, glyphs, energy_style, self.config, draw_outline=False)
        ax.grid(alpha=0.18)

    def _draw_time_panel(self, ax, hits, energy_style, z_limits):
        ax.set_ylabel("time [ns]")
        time_clusterer = TimeClusterer(
            gap_threshold_ns=self.config.time_gap_threshold_ns,
            bridge_threshold_ns=self.config.time_bridge_threshold_ns,
            gap_fraction=self.config.time_gap_fraction,
            min_display_gap_ns=self.config.time_gap_min_display_ns,
        )
        clusters = time_clusterer.cluster(hits)
        self.hit_renderer.draw_time_hits_by_energy(ax, hits, time_clusterer, clusters, energy_style, self.config)
        self._draw_time_cluster_breaks(ax, clusters, z_limits)
        self._set_time_limits(ax, clusters, z_limits)
        ax.grid(alpha=0.18)

    def _draw_energy_panel(self, ax, hits, energy_style, z_limits):
        ax.set_ylabel("energy [MeV]")
        self.hit_renderer.draw_energy_hits_by_energy(ax, hits, energy_style, self.config)
        ax.set_yscale("log")
        ax.set_xlim(z_limits)
        ax.grid(alpha=0.18)

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

    def _draw_time_cluster_breaks(self, ax, clusters, z_limits):
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

    def _set_spatial_limits(self, ax, glyphs, z_limits):
        coord_values = []
        for glyph in glyphs:
            coord_values.extend([glyph.rectangle.coord_min, glyph.rectangle.coord_max])
        ax.set_xlim(z_limits)
        ax.set_ylim(self._padded_limits(coord_values, default=(-8.0, 8.0)))

    def _set_time_limits(self, ax, clusters, z_limits):
        ax.set_xlim(z_limits)
        if clusters:
            y0 = min(cluster.display_start for cluster in clusters)
            y1 = max(cluster.display_stop for cluster in clusters)
            ax.set_ylim(self._padded_limits([y0, y1], default=(0.0, 1.0)))

    def _shared_z_limits(self, xz_glyphs, yz_glyphs, hits):
        z_values = []
        for glyph in xz_glyphs + yz_glyphs:
            z_values.extend([glyph.rectangle.z_min, glyph.rectangle.z_max])
        for hit in hits:
            if self._finite_display_value(hit.z):
                z_values.append(hit.z)
        return self._padded_limits(z_values, default=(0.0, 7.0))

    def _padded_limits(self, values, default):
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
