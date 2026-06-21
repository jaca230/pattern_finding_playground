from collections import Counter

import matplotlib.pyplot as plt

from figures.figure import PlotFigure
from figures.utils import particle_name_map
from models.event import Event


class MisclassificationPieFigure(PlotFigure):
    def __init__(self, max_labels=10, title=None, show: bool = True):
        super().__init__(show=show)
        self.max_labels = max_labels
        self.title = title

    def draw(self, events: list[Event], max_labels=None, title=None):
        max_labels = self.max_labels if max_labels is None else max_labels
        title = self.title if title is None else title

        sizes, legend_labels = self._error_groups(events, max_labels=max_labels)
        total = sum(sizes)

        fig, ax = plt.subplots(figsize=(10, 10))
        wedges, _, _ = ax.pie(
            sizes,
            labels=[f"Group {index + 1}" for index in range(len(sizes))],
            autopct=lambda p: f"{p:.1f}% ({int(round(p * total / 100))})" if p > 1 else "",
            textprops={"fontsize": 9},
            startangle=140,
        )

        ax.legend(wedges, legend_labels, title="Event Types", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)

        if title is None:
            title = f"Misclassified Events by Composition\n$N_{{\\text{{incorrect}}}} = {total}$"
        ax.set_title(title)

        plt.tight_layout()
        self._display(plt)
        return fig, ax

    def _error_groups(self, events, max_labels):
        error_groups = Counter()

        for event in events:
            if event.is_valid:
                continue

            info = event.extra_info.get("tracklet_algorithm_info", {})
            truth_counter = info.get("particles_in_event_truth", Counter())
            reco_counter = info.get("particles_in_event_reco", Counter())

            tracklets = {
                t
                for pattern in event.get_patterns()
                for vertex in pattern.get_vertices()
                for t in vertex.get_tracklets()
            }
            pattern_reco_counter = Counter(t.particle_id for t in tracklets)

            key = (
                frozenset(self._to_named(pattern_reco_counter).items()),
                frozenset(self._to_named(reco_counter).items()),
                frozenset(self._to_named(truth_counter).items()),
            )
            error_groups[key] += 1

        sizes = []
        legend_labels = []
        for (pattern_named, reco_named, truth_named), count in error_groups.items():
            sizes.append(count)
            label = (
                r"$\text{Pattern Particles: }" + self._particle_label(Counter(dict(pattern_named))) + r"$" + "\n"
                + r"$\text{Tracklet Particles: }" + self._particle_label(Counter(dict(reco_named))) + r"$" + "\n"
                + r"$\text{Truth Particles: }" + self._particle_label(Counter(dict(truth_named))) + r"$"
            )
            legend_labels.append(label)

        sorted_indices = sorted(range(len(sizes)), key=lambda index: sizes[index], reverse=True)
        sizes = [sizes[index] for index in sorted_indices]
        legend_labels = [legend_labels[index] for index in sorted_indices]

        if len(legend_labels) > max_labels:
            other_size = sum(sizes[max_labels:])
            sizes = sizes[:max_labels] + [other_size]
            legend_labels = legend_labels[:max_labels] + [r"$\text{Other}$"]

        return sizes, legend_labels

    def _to_named(self, counter):
        return Counter({
            particle_name_map.get(pid, particle_name_map["default"])["name"]: count
            for pid, count in counter.items()
        })

    def _particle_label(self, counter: Counter) -> str:
        return " + ".join(f"{v}{k}" if v > 1 else k for k, v in sorted(counter.items()))
