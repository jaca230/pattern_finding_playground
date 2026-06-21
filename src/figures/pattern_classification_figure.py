import warnings
from collections import Counter, defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from figures.figure import PlotFigure
from figures.utils import particle_name_map
from models.event import Event


class PatternClassificationFigure(PlotFigure):
    def __init__(self, particle_source="pattern_reco", title=None, show: bool = True):
        super().__init__(show=show)
        self.particle_source = particle_source
        self.title = title

    def draw(self, events: List[Event], use_truth_particles=None, title=None):
        particle_source = self._resolve_particle_source(use_truth_particles)
        title = self.title if title is None else title

        bins = self._group_events(events, particle_source)
        bin_labels, all_counts, correct_counts, failed_counts, correctness_percentages = self._prepare_bins(bins)

        fig, ax = plt.subplots(figsize=(10, 8))
        x = np.arange(len(bin_labels))
        width = 0.25

        ax.bar(x - width, all_counts, color="lightgray", label="Total", width=width)
        ax.bar(x, correct_counts, color="lightblue", label="Correct", width=width)
        ax.bar(x + width, failed_counts, color="lightcoral", label="Failed", width=width)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [r"$" + label + r"$" if label else r"$\emptyset$" for label in bin_labels],
            rotation=45,
            ha="right",
            fontsize=10,
        )
        ax.set_ylabel("Event Count")
        if title is None:
            title = f"Pattern Reconstruction by Particle Composition ({particle_source})\n" f"$N_{{\\text{{events}}}} = {len(events)}$"
        ax.set_title(title)
        ax.legend()
        ax.set_yscale("log")

        cmap = plt.get_cmap("coolwarm_r")
        norm = plt.Normalize(vmin=0, vmax=100)
        max_height = max(all_counts) if len(all_counts) else 1
        ax.set_ylim(0.9, max_height * 2)

        for index, percent in enumerate(correctness_percentages):
            color = cmap(norm(percent))
            text = f"{percent:.1f}%\n{correct_counts[index]} / {all_counts[index]}"
            ax.text(x[index], all_counts[index] + 0.05 * all_counts[index], text, ha="center", va="bottom", fontsize=9, color=color)

        plt.tight_layout()
        self._display(plt)
        return fig, ax

    def _resolve_particle_source(self, use_truth_particles):
        if use_truth_particles is None:
            return self.particle_source

        warnings.warn(
            "`use_truth_particles` is deprecated. Use `particle_source='truth'` instead.",
            DeprecationWarning,
        )
        if self.particle_source != "pattern_reco":
            raise ValueError("Cannot use both `particle_source` and `use_truth_particles`. Use only one.")
        return "truth" if use_truth_particles else "pattern_reco"

    def _group_events(self, events, particle_source):
        if particle_source not in {"truth", "reco", "pattern_reco"}:
            raise ValueError("particle_source must be one of {'truth', 'reco', 'pattern_reco'}")

        bins = defaultdict(list)
        for event in events:
            pid_counter = self._particle_counter(event, particle_source)
            particle_counts = Counter({
                particle_name_map.get(pid, particle_name_map["default"])["name"]: count
                for pid, count in pid_counter.items()
            })
            bins[frozenset(particle_counts.items())].append(event)
        return bins

    def _particle_counter(self, event, particle_source):
        if particle_source == "truth":
            return event.extra_info.get("tracklet_algorithm_info", {}).get("particles_in_event_truth", Counter())
        if particle_source == "reco":
            return event.extra_info.get("tracklet_algorithm_info", {}).get("particles_in_event_reco", Counter())

        tracklets = {
            t
            for pattern in event.get_patterns()
            for vertex in pattern.get_vertices()
            for t in vertex.get_tracklets()
        }
        return Counter(t.particle_id for t in tracklets)

    def _prepare_bins(self, bins):
        bin_labels = []
        all_counts = []
        correct_counts = []
        failed_counts = []
        correctness_percentages = []

        for particle_counter, grouped_events in bins.items():
            label = self._particle_label(Counter(dict(particle_counter)))
            total = len(grouped_events)
            correct = sum(1 for event in grouped_events if event.is_valid)
            failed = total - correct
            percentage = (correct / total) * 100 if total > 0 else 0

            bin_labels.append(label)
            all_counts.append(total)
            correct_counts.append(correct)
            failed_counts.append(failed)
            correctness_percentages.append(percentage)

        sorted_indices = np.argsort(all_counts)[::-1]
        return (
            [bin_labels[i] for i in sorted_indices],
            np.array(all_counts)[sorted_indices],
            np.array(correct_counts)[sorted_indices],
            np.array(failed_counts)[sorted_indices],
            np.array(correctness_percentages)[sorted_indices],
        )

    def _particle_label(self, counter: Counter) -> str:
        return " + ".join(f"{v}{k}" if v > 1 else k for k, v in sorted(counter.items()))
