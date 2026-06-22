from __future__ import annotations

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from figures.event_display.styles import PARTICLE_STYLES


class LegendRenderer:
    def draw_particle_legend(self, ax, particle_ids, include_vertex: bool = True, title: str | None = None) -> None:
        handles = []
        for particle_id in sorted(set(particle_ids), key=str):
            style = PARTICLE_STYLES.get(particle_id, PARTICLE_STYLES["default"])
            handles.append(
                Line2D(
                    [0, 1],
                    [0, 0],
                    color=style["color"],
                    linewidth=3.0,
                    marker="None",
                    label=f"${style['name']}$",
                )
            )

        if include_vertex:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="X",
                    color="magenta",
                    markeredgecolor="black",
                    linestyle="None",
                    label="vertex",
                )
            )
        if handles:
            ax.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                frameon=False,
                title=title,
            )

    def draw_energy_colorbar(self, fig, ax, energy_style) -> None:
        scalar = plt.cm.ScalarMappable(norm=energy_style.norm, cmap=energy_style.cmap)
        scalar.set_array([])
        colorbar = fig.colorbar(scalar, ax=ax, fraction=0.046, pad=0.02)
        colorbar.set_label("hit energy [MeV]")
