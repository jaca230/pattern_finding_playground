from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from figures.event_display.event_display_data import AggregatedStripGlyph
from figures.event_display.styles import particle_style


class HitRenderer:
    def draw_particle_spatial_hits(self, ax, glyphs, config) -> None:
        for glyph in self._aggregate_strips(glyphs):
            rect = glyph.rectangle
            mixed_color = self._mixed_particle_color(glyph.particle_counts)
            patch = Rectangle(
                (rect.z_min, rect.coord_min),
                rect.z_width,
                rect.coord_width,
                facecolor=mixed_color,
                edgecolor="none",
                linewidth=0.0,
                alpha=config.hit_alpha,
                zorder=4,
            )
            ax.add_patch(patch)
            self._draw_strip_count(ax, rect, glyph.hit_count, config)

    def draw_energy_spatial_hits(self, ax, glyphs, energy_style, config, draw_outline: bool = False) -> None:
        cmap = plt.get_cmap(energy_style.cmap)
        for glyph in glyphs:
            rect = glyph.rectangle
            facecolor = cmap(energy_style.norm(max(glyph.energy, config.energy_floor_mev)))
            edgecolor = "none" if not draw_outline else config.hit_edge_color
            patch = Rectangle(
                (rect.z_min, rect.coord_min),
                rect.z_width,
                rect.coord_width,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=config.hit_edge_width,
                alpha=config.hit_alpha,
                zorder=4,
            )
            if draw_outline:
                patch.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=config.hit_edge_width + 0.75, foreground="black"),
                        path_effects.Normal(),
                    ]
                )
            ax.add_patch(patch)

    def draw_time_hits(self, ax, hits, time_clusterer, clusters, config) -> None:
        for hit in hits:
            display_time = time_clusterer.display_time(hit, clusters)
            if display_time is None:
                continue
            color = particle_style(hit.particle_id)["color"]
            ax.scatter(
                hit.z,
                display_time,
                s=32,
                c=[color],
                edgecolors=config.hit_edge_color,
                linewidths=0.35,
                zorder=4,
            )

    def draw_time_hits_by_energy(self, ax, hits, time_clusterer, clusters, energy_style, config) -> None:
        cmap = plt.get_cmap(energy_style.cmap)
        for hit in hits:
            display_time = time_clusterer.display_time(hit, clusters)
            if display_time is None:
                continue
            energy = max(hit.energy or config.energy_floor_mev, config.energy_floor_mev)
            ax.scatter(
                hit.z,
                display_time,
                s=32,
                c=[cmap(energy_style.norm(energy))],
                edgecolors="none",
                linewidths=0.0,
                zorder=4,
            )

    def draw_energy_hits(self, ax, hits, config) -> None:
        for hit in hits:
            if hit.z is None or hit.energy is None:
                continue
            energy = max(hit.energy, config.energy_floor_mev)
            color = particle_style(hit.particle_id)["color"]
            ax.scatter(
                hit.z,
                energy,
                s=30,
                c=[color],
                edgecolors=config.hit_edge_color,
                linewidths=0.35,
                zorder=4,
            )

    def draw_energy_hits_by_energy(self, ax, hits, energy_style, config) -> None:
        cmap = plt.get_cmap(energy_style.cmap)
        for hit in hits:
            if hit.z is None or hit.energy is None:
                continue
            energy = max(hit.energy, config.energy_floor_mev)
            ax.scatter(
                hit.z,
                energy,
                s=30,
                c=[cmap(energy_style.norm(energy))],
                edgecolors="none",
                linewidths=0.0,
                zorder=4,
            )

    def draw_layer_guides(self, ax, rectangles, config) -> None:
        y0, y1 = ax.get_ylim()
        for rect in rectangles:
            ax.add_patch(
                Rectangle(
                    (rect.z_min, y0),
                    rect.z_width,
                    y1 - y0,
                    facecolor="0.5",
                    edgecolor="none",
                    alpha=config.geometry_guide_alpha,
                    zorder=1,
                )
            )

    def _aggregate_strips(self, glyphs):
        aggregated = {}
        for glyph in glyphs:
            key = (
                glyph.plane,
                glyph.hit.volume_id,
                round(glyph.rectangle.z_min, 6),
                round(glyph.rectangle.coord_min, 6),
                round(glyph.rectangle.z_width, 6),
                round(glyph.rectangle.coord_width, 6),
            )
            if key not in aggregated:
                aggregated[key] = {
                    "plane": glyph.plane,
                    "rectangle": glyph.rectangle,
                    "particle_counts": Counter(),
                    "hit_count": 0,
                    "total_energy": 0.0,
                }
            aggregated[key]["particle_counts"][int(glyph.particle_id)] += 1
            aggregated[key]["hit_count"] += 1
            aggregated[key]["total_energy"] += float(glyph.energy or 0.0)
        return [
            AggregatedStripGlyph(
                plane=values["plane"],
                rectangle=values["rectangle"],
                particle_counts=values["particle_counts"],
                hit_count=values["hit_count"],
                total_energy=values["total_energy"],
            )
            for values in aggregated.values()
        ]

    def _mixed_particle_color(self, particle_counts) -> tuple[float, float, float]:
        total = sum(particle_counts.values())
        if total <= 0:
            return to_rgb("#999999")

        mixed = [0.0, 0.0, 0.0]
        for particle_id, count in particle_counts.items():
            rgb = to_rgb(particle_style(particle_id)["color"])
            weight = count / total
            for index in range(3):
                mixed[index] += weight * rgb[index]
        return tuple(mixed)

    def _draw_strip_count(self, ax, rect, hit_count: int, config) -> None:
        ax.text(
            rect.z_min + 0.5 * rect.z_width,
            rect.coord_min + 0.5 * rect.coord_width,
            str(hit_count),
            color="white",
            fontsize=config.strip_count_fontsize,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
            path_effects=[path_effects.Stroke(linewidth=1.2, foreground="black"), path_effects.Normal()],
        )
