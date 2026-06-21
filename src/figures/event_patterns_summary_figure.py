import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from figures.figure import PlotFigure
from models.event import Event


class EventPatternsSummaryFigure(PlotFigure):
    def __init__(self, title=None, show: bool = True):
        super().__init__(show=show)
        self.title = title

    def draw(self, events: list[Event], title=None):
        title = self.title if title is None else title
        n_truth, n_reco, passed, correct_count = self._event_metrics(events)

        max_val = max(max(n_truth, default=0), max(n_reco, default=0))
        bins = np.arange(0, max_val + 2)

        fig, ax = plt.subplots(2, 2, figsize=(16, 12))

        self._draw_truth_vs_reco_hist(ax[0, 0], n_truth, n_reco, bins)
        im2d = self._draw_2d_truth_vs_reco(ax[0, 1], n_truth, n_reco, bins)
        fig.colorbar(im2d, ax=ax[0, 1], label="Number of Events")

        conf_matrix, im_conf = self._draw_confusion_matrix(
            ax[1, 0],
            passed,
            correct_count,
            len(events),
            x_axis_label="Pattern Count Correct",
            y_axis_label="Passed Validation",
        )
        fig.colorbar(im_conf, ax=ax[1, 0], label="Event Count")

        self._draw_info_box(ax[1, 1], conf_matrix, len(events))

        fig.suptitle(title, fontsize=20)
        plt.tight_layout()
        self._display(plt)
        return fig, ax

    def _event_metrics(self, events):
        n_truth = []
        n_reco = []
        passed = []
        correct_count = []

        for event in events:
            truth_count = event.extra_info.get("tracklet_algorithm_info", {}).get("n_patterns_truth", 0)
            reco_count = len(event.get_patterns())
            valid = event.is_valid
            count_match = reco_count == truth_count

            n_truth.append(truth_count)
            n_reco.append(reco_count)
            passed.append(valid)
            correct_count.append(count_match)

        return n_truth, n_reco, passed, correct_count

    def _draw_truth_vs_reco_hist(self, ax, n_truth, n_reco, bins):
        ax.hist(n_truth, bins=bins, color="tab:blue", alpha=0.7, label="Truth Patterns")
        ax.hist(n_reco, bins=bins, color="tab:red", alpha=0.7, label="Reco Patterns")
        ax.set_xlabel("Number of Patterns", fontsize=18)
        ax.set_ylabel("Events", fontsize=18)
        ax.legend(fontsize=16)
        ax.grid(True)
        ax.set_xticks(bins[:-1] + 0.5)
        ax.set_xticklabels(bins[:-1], fontsize=16)
        ax.tick_params(axis="y", labelsize=16)

    def _draw_2d_truth_vs_reco(self, ax, n_truth, n_reco, bins):
        counts, xedges, yedges, im = ax.hist2d(n_truth, n_reco, bins=[bins, bins], cmap="viridis", norm=LogNorm())
        for i in range(len(xedges) - 1):
            for j in range(len(yedges) - 1):
                count = counts[i, j]
                if count > 0:
                    x_pos = (xedges[i] + xedges[i + 1]) / 2
                    y_pos = (yedges[j] + yedges[j + 1]) / 2
                    ax.text(x_pos, y_pos, f"{int(count)}", ha="center", va="center", color="black", fontweight="bold", fontsize=14)

        ax.set_xlabel("Truth Patterns", fontsize=18)
        ax.set_ylabel("Reco Patterns", fontsize=18)
        ax.set_xticks(bins[:-1] + 0.5)
        ax.set_yticks(bins[:-1] + 0.5)
        ax.set_xticklabels(bins[:-1], fontsize=16)
        ax.set_yticklabels(bins[:-1], fontsize=16)
        return im

    def _draw_confusion_matrix(
        self,
        ax,
        metric1,
        metric2,
        total,
        x_axis_label="Metric 2 (X)",
        y_axis_label="Metric 1 (Y)",
        x_tick_labels=("False", "True"),
        y_tick_labels=("True", "False"),
    ):
        conf_matrix = np.zeros((2, 2), dtype=int)
        for m1, m2 in zip(metric1, metric2):
            row = 0 if m1 else 1
            col = 1 if m2 else 0
            conf_matrix[row, col] += 1

        im = ax.imshow(conf_matrix, cmap="Blues", interpolation="nearest", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(x_tick_labels, fontsize=16)
        ax.set_yticklabels(y_tick_labels, fontsize=16)
        ax.set_xlabel(x_axis_label, fontsize=18)
        ax.set_ylabel(y_axis_label, fontsize=18)

        for i in range(2):
            for j in range(2):
                count = conf_matrix[i, j]
                ax.text(j, i, f"{count} ({count/total:.2%})", ha="center", va="center", color="orange", fontweight="bold", fontsize=14)

        return conf_matrix, im

    def _draw_info_box(self, ax, conf_matrix, total):
        n_failed = int(conf_matrix[1, 0] + conf_matrix[1, 1])
        n_passed = int(conf_matrix[0, 0] + conf_matrix[0, 1])
        performance = (conf_matrix[0, 1]) / total if total > 0 else 0

        ax.axis("off")
        ax.text(0.1, 0.8, f"Total Events: {total}", fontsize=16)
        ax.text(0.1, 0.7, f"Passed Validation: {n_passed}", fontsize=16)
        ax.text(0.1, 0.6, f"Failed Validation: {n_failed}", fontsize=16)
        ax.text(0.1, 0.5, f"Performance: {performance:.2%}", fontsize=16)
