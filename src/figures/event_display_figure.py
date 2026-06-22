import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from figures.figure import PlotFigure
from figures.utils import particle_style
from models.event import Event


class EventDisplayFigure(PlotFigure):
    def draw(self, event: Event):
        tracklets = self._tracklets_for_event(event)

        fig, ax = plt.subplots(
            4,
            1,
            figsize=(12, 10),
            sharex=True,
            gridspec_kw={"hspace": 0.0},
        )

        self._draw_x_view(ax[0], tracklets)
        self._draw_y_view(ax[1], tracklets)
        self._draw_time_view(ax[2], tracklets)
        self._draw_energy_view(ax[3], tracklets)
        self._draw_centroids([ax[0], ax[1]], event)

        for axis in ax:
            axis.set_xlim(-0.5, 7)
        for axis in ax[:-1]:
            axis.label_outer()

        ax[3].legend(title="Tracklet Types", loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=10)

        fig.align_ylabels(ax)
        plt.subplots_adjust(right=0.75)
        plt.tight_layout()
        self._display(plt)

        return fig, ax

    def _tracklets_for_event(self, event: Event):
        if event.get_patterns():
            return {t for pattern in event.get_patterns() for t in pattern.get_unique_tracklets()}
        return event.all_tracklets

    def _darken_color(self, color: str, factor: float = 0.6) -> str:
        color = np.array(mcolors.hex2color(color))
        darkened_color = np.clip(color * factor, 0, 1)
        return mcolors.rgb2hex(darkened_color)

    def _tracklet_style(self, tracklet):
        return particle_style(tracklet.particle_id)

    def _draw_dashed_lines(self, ax, dz_total=(0.139 + 0.149), start=0, step=2, y_ranges=None):
        step_size = dz_total / 2.0
        z_positions = np.arange(start, 7, step_size * step)

        if y_ranges is None:
            for z in z_positions:
                ax.axvline(x=z, color="gray", linestyle="--", alpha=0.5, linewidth=1.0)
            return

        y_min_data, y_max_data = ax.get_ylim()
        for z in z_positions:
            for ymin, ymax in y_ranges:
                ymin_frac = (ymin - y_min_data) / (y_max_data - y_min_data)
                ymax_frac = (ymax - y_min_data) / (y_max_data - y_min_data)
                ax.axvline(
                    x=z,
                    color="gray",
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1.0,
                    ymin=ymin_frac,
                    ymax=ymax_frac,
                )

    def _draw_x_view(self, ax, tracklets):
        ax.set_ylabel("x [mm]")
        self._draw_dashed_lines(ax, start=0, step=2)

        for tracklet in tracklets:
            style = self._tracklet_style(tracklet)
            color = style["color"]
            front_hits = tracklet.get_front_hits()
            ax.scatter(
                [hit.z for hit in front_hits],
                [hit.x for hit in front_hits],
                color=color,
                alpha=0.7,
                label=f"${style['name']}$",
            )

            ep0, ep1 = tracklet.get_endpoints()
            if ep0 and ep1:
                ax.scatter([ep0.z, ep1.z], [ep0.x, ep1.x], color=color, s=150, marker="*", zorder=5, alpha=0.4)
                ax.plot(
                    [ep0.z, ep1.z],
                    [ep0.x, ep1.x],
                    color=self._darken_color(color, 0.3),
                    linestyle="-",
                    linewidth=3,
                    alpha=0.7,
                )

    def _draw_y_view(self, ax, tracklets):
        ax.set_ylabel("y [mm]")
        self._draw_dashed_lines(ax, start=(0.139 + 0.149) / 2, step=2)

        for tracklet in tracklets:
            style = self._tracklet_style(tracklet)
            color = style["color"]
            back_hits = tracklet.get_back_hits()
            ax.scatter(
                [hit.z for hit in back_hits],
                [hit.y for hit in back_hits],
                color=color,
                alpha=0.7,
                label=f"${style['name']}$",
            )

            ep0, ep1 = tracklet.get_endpoints()
            if ep0 and ep1:
                ax.scatter([ep0.z, ep1.z], [ep0.y, ep1.y], color=color, s=150, marker="*", zorder=5, alpha=0.4)
                ax.plot(
                    [ep0.z, ep1.z],
                    [ep0.y, ep1.y],
                    color=self._darken_color(color, 0.3),
                    linestyle="-",
                    linewidth=3,
                    alpha=0.7,
                )

    def _draw_time_view(self, ax, tracklets, time_gap_threshold=0.3, gap_percentage=0.1):
        ax.set_ylabel("Time [ns]")

        all_hits = []
        for tracklet in tracklets:
            color = self._tracklet_style(tracklet)["color"]
            for hit in tracklet.hits:
                all_hits.append((hit, color))
        all_hits.sort(key=lambda pair: pair[0].time)

        if not all_hits:
            return

        grouped_hits = []
        group = [all_hits[0]]
        for prev, curr in zip(all_hits, all_hits[1:]):
            if abs(curr[0].time - prev[0].time) > time_gap_threshold:
                grouped_hits.append(group)
                group = [curr]
            else:
                group.append(curr)
        grouped_hits.append(group)

        normalized_times = []
        group_min_times, group_max_times = [], []
        cumulative_offset = 0

        for group in grouped_hits:
            t0 = min(hit.time for hit, _ in group)
            t1 = max(hit.time for hit, _ in group)
            normalized_times.extend([hit.time - t0 + cumulative_offset for hit, _ in group])
            group_min_times.append(t0 + cumulative_offset)
            group_max_times.append(t1 + cumulative_offset)
            cumulative_offset += t1 - t0

        total_range = max(normalized_times) - min(normalized_times)
        gap_size = total_range * gap_percentage

        normalized_times_with_gap = []
        adjusted_group_ends = []
        cumulative_offset = 0

        for index, group in enumerate(grouped_hits):
            t0 = min(hit.time for hit, _ in group)
            t1 = max(hit.time for hit, _ in group)
            normalized_times_with_gap.extend([hit.time - t0 + cumulative_offset for hit, _ in group])
            adjusted_group_ends.append((cumulative_offset, cumulative_offset + t1 - t0))
            cumulative_offset += (t1 - t0) + gap_size

            if index < len(grouped_hits) - 1:
                z_min, z_max = ax.get_xlim()
                ax.fill_betweenx([cumulative_offset - gap_size, cumulative_offset], z_min, z_max, color="gray", alpha=0.2, hatch="//")

        zs = [hit.z for hit, _ in all_hits]
        colors = [color for _, color in all_hits]

        ax.scatter(zs, normalized_times_with_gap, c=colors, alpha=0.7)
        ax.set_xlim([0, max(zs) + 2])

        y_ticks = [y for bounds in adjusted_group_ends for y in bounds]
        y_labels = [f"{t:.2f}" for pair in zip(group_min_times, group_max_times) for t in pair]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

        t_min, t_max = ax.get_ylim()
        adjusted_group_ends[0] = (t_min, adjusted_group_ends[0][1])
        adjusted_group_ends[-1] = (adjusted_group_ends[-1][0], t_max)

        self._draw_dashed_lines(ax, start=0, step=1, y_ranges=adjusted_group_ends)

    def _draw_energy_view(self, ax, tracklets):
        ax.set_ylabel("Energy [MeV]")
        self._draw_dashed_lines(ax, start=0, step=1)

        for tracklet in tracklets:
            style = self._tracklet_style(tracklet)
            color = style["color"]
            ax.scatter(
                [hit.z for hit in tracklet.hits],
                [hit.energy for hit in tracklet.hits],
                color=color,
                alpha=0.7,
                label=f"${style['name']}$",
            )

        ax.set_yscale("log")
        ax.set_xlabel("z [mm]")

    def _draw_centroids(self, ax, event):
        if "vertex_algorithm_info" not in event.extra_info:
            return

        vertex_algorithm_info = event.extra_info["vertex_algorithm_info"]
        if "stats" not in vertex_algorithm_info:
            return

        stats = vertex_algorithm_info["stats"]

        if "front" in stats and "centroids" in stats["front"]:
            for centroid in stats["front"]["centroids"]:
                ax[0].scatter(
                    centroid[2],
                    centroid[0],
                    color="magenta",
                    edgecolors="black",
                    marker="X",
                    s=150,
                    linewidths=1.5,
                    zorder=15,
                    label="Front Centroid",
                )

        if "back" in stats and "centroids" in stats["back"]:
            for centroid in stats["back"]["centroids"]:
                ax[1].scatter(
                    centroid[2],
                    centroid[1],
                    color="magenta",
                    edgecolors="black",
                    marker="X",
                    s=150,
                    linewidths=1.5,
                    zorder=15,
                    label="Back Centroid",
                )
