import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
from collections import Counter, defaultdict
from models.event_patterns import EventPatterns
from utils.particle_mapping import particle_name_map
import warnings
from typing import List



def darken_color(color: str, factor: float = 0.6) -> str:
    color = np.array(mcolors.hex2color(color))
    darkened_color = np.clip(color * factor, 0, 1)
    return mcolors.rgb2hex(darkened_color)


def draw_dashed_lines(ax, dz_total=(0.139 + 0.149), start=0, step=2, y_ranges=None):
    """
    Draw dashed vertical lines along z-axis positions.
    Optionally restrict drawing to given `y_ranges` (list of (ymin, ymax) in data coordinates).
    """
    step_size = dz_total / 2.0
    z_positions = np.arange(start, 7, step_size * step)

    if y_ranges is None:
        for z in z_positions:
            ax.axvline(x=z, color='gray', linestyle='--', alpha=0.5, linewidth=1.0)
    else:
        y_min_data, y_max_data = ax.get_ylim()
        for z in z_positions:
            for ymin, ymax in y_ranges:
                ymin_frac = (ymin - y_min_data) / (y_max_data - y_min_data)
                ymax_frac = (ymax - y_min_data) / (y_max_data - y_min_data)
                ax.axvline(
                    x=z,
                    color='gray',
                    linestyle='--',
                    alpha=0.5,
                    linewidth=1.0,
                    ymin=ymin_frac,
                    ymax=ymax_frac
                )


def plot_x_view(ax, tracklets):
    ax.set_ylabel("x [mm]")
    draw_dashed_lines(ax, start=0, step=2)

    for tracklet in tracklets:
        color = tracklet.particle_color
        front_hits = tracklet.get_front_hits()
        ax.scatter([hit.z for hit in front_hits],
                   [hit.x for hit in front_hits],
                   color=color,
                   alpha=0.7,
                   label=f"${tracklet.particle_name}$")

        ep0, ep1 = tracklet.get_endpoints()
        if ep0 and ep1:
            ax.scatter([ep0.z, ep1.z], [ep0.x, ep1.x], color=color, s=150, marker='*', zorder=5, alpha=0.4)
            ax.plot([ep0.z, ep1.z], [ep0.x, ep1.x], color=darken_color(color, 0.3), linestyle='-', linewidth=3, alpha=0.7)


def plot_y_view(ax, tracklets):
    ax.set_ylabel("y [mm]")
    draw_dashed_lines(ax, start=(0.139 + 0.149) / 2, step=2)

    for tracklet in tracklets:
        color = tracklet.particle_color
        back_hits = tracklet.get_back_hits()
        ax.scatter([hit.z for hit in back_hits],
                   [hit.y for hit in back_hits],
                   color=color,
                   alpha=0.7,
                   label=f"${tracklet.particle_name}$")

        ep0, ep1 = tracklet.get_endpoints()
        if ep0 and ep1:
            ax.scatter([ep0.z, ep1.z], [ep0.y, ep1.y], color=color, s=150, marker='*', zorder=5, alpha=0.4)
            ax.plot([ep0.z, ep1.z], [ep0.y, ep1.y], color=darken_color(color, 0.3), linestyle='-', linewidth=3, alpha=0.7)


def plot_time_view(ax, tracklets, time_gap_threshold=0.2, gap_percentage=0.1):
    ax.set_ylabel("Time [ns]")

    all_hits = sorted([hit for t in tracklets for hit in t.hits], key=lambda h: h.time)

    # Group hits by time
    grouped_hits = []
    group = [all_hits[0]]
    for prev, curr in zip(all_hits, all_hits[1:]):
        if abs(curr.time - prev.time) > time_gap_threshold:
            grouped_hits.append(group)
            group = [curr]
        else:
            group.append(curr)
    grouped_hits.append(group)

    # Normalize times and apply gap
    normalized_times = []
    group_min_times, group_max_times = [], []
    cumulative_offset = 0

    for group in grouped_hits:
        t0 = min(hit.time for hit in group)
        t1 = max(hit.time for hit in group)
        normalized_times.extend([hit.time - t0 + cumulative_offset for hit in group])
        group_min_times.append(t0 + cumulative_offset)
        group_max_times.append(t1 + cumulative_offset)
        cumulative_offset += t1 - t0

    total_range = max(normalized_times) - min(normalized_times)
    gap_size = total_range * gap_percentage

    normalized_times_with_gap = []
    adjusted_group_ends = []
    cumulative_offset = 0

    for i, group in enumerate(grouped_hits):
        t0 = min(hit.time for hit in group)
        t1 = max(hit.time for hit in group)
        normalized_times_with_gap.extend([hit.time - t0 + cumulative_offset for hit in group])
        adjusted_group_ends.append((cumulative_offset, cumulative_offset + t1 - t0))
        cumulative_offset += (t1 - t0) + gap_size

        if i < len(grouped_hits) - 1:
            z_min, z_max = ax.get_xlim()
            ax.fill_betweenx([cumulative_offset - gap_size, cumulative_offset],
                             z_min, z_max, color='gray', alpha=0.2, hatch='//')

    zs = [hit.z for hit in all_hits]
    colors = [hit.particle_color for hit in all_hits]
    ax.scatter(zs, normalized_times_with_gap, c=colors, alpha=0.7)
    ax.set_xlim([0, max(zs) + 2])

    # Label y-axis with original times
    y_ticks = [y for bounds in adjusted_group_ends for y in bounds]
    y_labels = [f"{t:.2f}" for pair in zip(group_min_times, group_max_times) for t in pair]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Extend top and bottom y-ranges for dashed line drawing
    t_min, t_max = ax.get_ylim()
    adjusted_group_ends[0] = (t_min, adjusted_group_ends[0][1])
    adjusted_group_ends[-1] = (adjusted_group_ends[-1][0], t_max)

    draw_dashed_lines(ax, start=0, step=1, y_ranges=adjusted_group_ends)


def plot_energy_view(ax, tracklets):
    ax.set_ylabel("Energy [MeV]")
    draw_dashed_lines(ax, start=0, step=1)

    for tracklet in tracklets:
        color = tracklet.particle_color
        front_hits = tracklet.get_front_hits()
        ax.scatter([hit.z for hit in front_hits],
                   [hit.energy for hit in front_hits],
                   color=color,
                   alpha=0.7,
                   label=f"${tracklet.particle_name}$")
    
    ax.set_yscale('log')
    ax.set_xlabel("z [mm]")


def plot_centroids(ax, event_patterns):
    """
    Plot centroids for the front and back views if available in event_patterns.
    The centroids are shown in magenta with black edges.
    """
    if 'vertex_algorithm_info' in event_patterns.extra_info:
        vertex_algorithm_info = event_patterns.extra_info['vertex_algorithm_info']

        if 'stats' in vertex_algorithm_info:
            stats = vertex_algorithm_info['stats']

            if 'front' in stats and 'centroids' in stats['front']:
                front_centroids = stats['front']['centroids']
                for centroid in front_centroids:
                    ax[0].scatter(
                        centroid[2], centroid[0],
                        color='magenta', edgecolors='black', marker='X',
                        s=150, linewidths=1.5, zorder=15, label='Front Centroid'
                    )

            if 'back' in stats and 'centroids' in stats['back']:
                back_centroids = stats['back']['centroids']
                for centroid in back_centroids:
                    ax[1].scatter(
                        centroid[2], centroid[1],
                        color='magenta', edgecolors='black', marker='X',
                        s=150, linewidths=1.5, zorder=15, label='Back Centroid'
                    )


def plot_event(event_patterns: EventPatterns):
    tracklets = {t for pattern in event_patterns.get_patterns() for t in pattern.get_unique_tracklets()}

    fig, ax = plt.subplots(
        4, 1, figsize=(12, 10),
        sharex=True,
        gridspec_kw={"hspace": 0.0}
    )

    plot_x_view(ax[0], tracklets)
    plot_y_view(ax[1], tracklets)
    plot_time_view(ax[2], tracklets)
    plot_energy_view(ax[3], tracklets)

    plot_centroids([ax[0], ax[1]], event_patterns)

    for a in ax:
        a.set_xlim(-0.5, 7)  # Set z-limits
    for a in ax[:-1]:
        a.label_outer()

    ax[3].legend(title="Tracklet Types", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

    fig.align_ylabels(ax)
    plt.subplots_adjust(right=0.75)
    plt.tight_layout()
    plt.show()

    return fig, ax


def generate_particle_label(counter: Counter) -> str:
    """Generate a readable label from a Counter of particle names."""
    return ' + '.join(f'{v}{k}' if v > 1 else k for k, v in sorted(counter.items()))

def plot_event_classification_from_patterns(event_patterns_list, particle_source="pattern_reco", title=None, use_truth_particles=None):
    """
    Classify events based on particle composition and plot validation success.

    Parameters:
    - event_patterns_list: list of EventPatterns
    - particle_source: one of {"truth", "reco", "pattern_reco"}
    - title: optional plot title
    - use_truth_particles: deprecated, use `particle_source='truth'` instead
    """
    # Handle deprecated use_truth_particles keyword
    if use_truth_particles is not None:
        warnings.warn(
            "`use_truth_particles` is deprecated. Use `particle_source='truth'` instead.",
            DeprecationWarning
        )
        # If particle_source is not already set, use the deprecated value
        if particle_source != "pattern_reco":
            raise ValueError("Cannot use both `particle_source` and `use_truth_particles`. Use only one.")
        particle_source = "truth" if use_truth_particles else "pattern_reco"

    # Validate the particle_source
    assert particle_source in {"truth", "reco", "pattern_reco"}, "Invalid particle_source"

    bins = defaultdict(list)

    for ep in event_patterns_list:
        if particle_source == "truth":
            pid_counter = ep.extra_info.get("tracklet_algorithm_info", {}).get("particles_in_event_truth", Counter())
        elif particle_source == "reco":
            pid_counter = ep.extra_info.get("tracklet_algorithm_info", {}).get("particles_in_event_reco", Counter())
        else:  # "pattern_reco"
            tracklets = {
                t for pattern in ep.get_patterns()
                for vertex in pattern.get_vertices()
                for t in vertex.get_tracklets()
            }
            pid_counter = Counter(t.particle_id for t in tracklets)

        particle_counts = Counter({
            particle_name_map.get(pid, particle_name_map["default"])["name"]: count
            for pid, count in pid_counter.items()
        })

        bins[frozenset(particle_counts.items())].append(ep)

    # Prepare plotting bins
    bin_labels = []
    all_counts = []
    correct_counts = []
    failed_counts = []
    correctness_percentages = []

    for particle_counter, eps in bins.items():
        label = generate_particle_label(Counter(dict(particle_counter)))
        total = len(eps)
        correct = sum(1 for ep in eps if ep.validate())
        failed = total - correct
        percentage = (correct / total) * 100 if total > 0 else 0

        bin_labels.append(label)
        all_counts.append(total)
        correct_counts.append(correct)
        failed_counts.append(failed)
        correctness_percentages.append(percentage)

    # Sort bins by total
    sorted_indices = np.argsort(all_counts)[::-1]
    bin_labels = [bin_labels[i] for i in sorted_indices]
    all_counts = np.array(all_counts)[sorted_indices]
    correct_counts = np.array(correct_counts)[sorted_indices]
    failed_counts = np.array(failed_counts)[sorted_indices]
    correctness_percentages = np.array(correctness_percentages)[sorted_indices]

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 8))
    x = np.arange(len(bin_labels))
    width = 0.25

    ax1.bar(x - width, all_counts, color='lightgray', label="Total", width=width)
    ax1.bar(x, correct_counts, color='lightblue', label="Correct", width=width)
    ax1.bar(x + width, failed_counts, color='lightcoral', label="Failed", width=width)

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [r"$" + label + r"$" if label else r"$\emptyset$" for label in bin_labels],
        rotation=45, ha="right", fontsize=10
    )
    ax1.set_ylabel("Event Count")
    if title is None:
        title = f"Pattern Reconstruction by Particle Composition ({particle_source})\n" \
                f"$N_{{\\text{{events}}}} = {len(event_patterns_list)}$"
    ax1.set_title(title)
    ax1.legend()
    ax1.set_yscale('log')

    cmap = plt.get_cmap("coolwarm_r")
    norm = plt.Normalize(vmin=0, vmax=100)
    max_height = max(all_counts)
    ax1.set_ylim(0.9, max_height * 2)

    for i, pct in enumerate(correctness_percentages):
        color = cmap(norm(pct))
        text = f"{pct:.1f}%\n{correct_counts[i]} / {all_counts[i]}"
        ax1.text(x[i], all_counts[i] + 0.05 * all_counts[i], text,
                 ha='center', va='bottom', fontsize=9, color=color)

    plt.tight_layout()
    plt.show()

def plot_truth_vs_reco_hist(ax, n_truth, n_reco, bins):
    ax.hist(n_truth, bins=bins, color='tab:blue', alpha=0.7, label='Truth Patterns')
    ax.hist(n_reco, bins=bins, color='tab:red', alpha=0.7, label='Reco Patterns')
    ax.set_xlabel('Number of Patterns', fontsize=18)
    ax.set_ylabel('Events', fontsize=18)
    ax.legend(fontsize=16)
    ax.grid(True)
    ax.set_xticks(bins[:-1] + 0.5)
    ax.set_xticklabels(bins[:-1], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)

def plot_2d_truth_vs_reco(ax, n_truth, n_reco, bins):
    counts, xedges, yedges, im = ax.hist2d(n_truth, n_reco, bins=[bins, bins], cmap='viridis', norm=LogNorm())
    for i in range(len(xedges) - 1):
        for j in range(len(yedges) - 1):
            count = counts[i, j]
            if count > 0:
                x_pos = (xedges[i] + xedges[i + 1]) / 2
                y_pos = (yedges[j] + yedges[j + 1]) / 2
                ax.text(x_pos, y_pos, f'{int(count)}', ha='center', va='center', color='black', fontweight='bold', fontsize=14)

    ax.set_xlabel('Truth Patterns', fontsize=18)
    ax.set_ylabel('Reco Patterns', fontsize=18)
    ax.set_xticks(bins[:-1] + 0.5)
    ax.set_yticks(bins[:-1] + 0.5)
    ax.set_xticklabels(bins[:-1], fontsize=16)
    ax.set_yticklabels(bins[:-1], fontsize=16)
    return im

def plot_confusion_matrix(ax, metric1, metric2, total, 
                          x_axis_label="Metric 2 (X)", 
                          y_axis_label="Metric 1 (Y)", 
                          x_tick_labels=('False', 'True'), 
                          y_tick_labels=('True', 'False')):
    conf_matrix = np.zeros((2, 2), dtype=int)
    for m1, m2 in zip(metric1, metric2):
        row = 0 if m1 else 1
        col = 1 if m2 else 0
        conf_matrix[row, col] += 1

    im = ax.imshow(conf_matrix, cmap='Blues', interpolation='nearest', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(x_tick_labels, fontsize=16)
    ax.set_yticklabels(y_tick_labels, fontsize=16)
    ax.set_xlabel(x_axis_label, fontsize=18)
    ax.set_ylabel(y_axis_label, fontsize=18)

    for i in range(2):
        for j in range(2):
            count = conf_matrix[i, j]
            ax.text(j, i, f'{count} ({count/total:.2%})', 
                    ha='center', va='center', color='orange', fontweight='bold', fontsize=14)

    return conf_matrix, im


def plot_info_box(ax, conf_matrix, total):
    n_failed = int(conf_matrix[1, 0] + conf_matrix[1, 1])
    n_passed = int(conf_matrix[0, 0] + conf_matrix[0, 1])
    performance = (conf_matrix[0, 1]) / total if total > 0 else 0

    ax.axis('off')
    ax.text(0.1, 0.8, f'Total Events: {total}', fontsize=16)
    ax.text(0.1, 0.7, f'Passed Validation: {n_passed}', fontsize=16)
    ax.text(0.1, 0.6, f'Failed Validation: {n_failed}', fontsize=16)
    ax.text(0.1, 0.5, f'Performance: {performance:.2%}', fontsize=16)


def plot_event_patterns_summary(events: List[EventPatterns], title: str):
    n_truth = []
    n_reco = []
    passed = []
    correct_count = []

    for event in events:
        truth_count = event.extra_info.get("tracklet_algorithm_info", {}).get("n_patterns_truth", 0)
        reco_count = len(event.get_patterns())
        valid = event.validate()
        count_match = reco_count == truth_count

        n_truth.append(truth_count)
        n_reco.append(reco_count)
        passed.append(valid)
        correct_count.append(count_match)

    max_val = max(max(n_truth, default=0), max(n_reco, default=0))
    bins = np.arange(0, max_val + 2)

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))

    plot_truth_vs_reco_hist(ax[0, 0], n_truth, n_reco, bins)
    im2d = plot_2d_truth_vs_reco(ax[0, 1], n_truth, n_reco, bins)
    fig.colorbar(im2d, ax=ax[0, 1], label='Number of Events')

    conf_matrix, im_conf = plot_confusion_matrix(
        ax[1, 0],
        passed,                   # Metric 1 (Y-axis)
        correct_count,            # Metric 2 (X-axis)
        len(events),
        x_axis_label="Pattern Count Correct",
        y_axis_label="Passed Validation"
    )
    fig.colorbar(im_conf, ax=ax[1, 0], label='Event Count')

    plot_info_box(ax[1, 1], conf_matrix, len(events))

    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()

def plot_incorrect_event_misclassification_pie(event_patterns_list, max_labels=10, title=None):
    """
    Plots a pie chart of incorrect event classifications.
    Each slice corresponds to a unique (pattern_reco, reco, truth) particle composition triplet.

    Parameters:
    - event_patterns_list: list of EventPatterns
    - max_labels: maximum number of pie slices to label explicitly
    - title: optional title for the plot
    """
    import matplotlib.pyplot as plt
    from collections import Counter

    error_groups = Counter()

    for ep in event_patterns_list:
        if ep.validate():
            continue

        info = ep.extra_info.get("tracklet_algorithm_info", {})
        truth_counter = info.get("particles_in_event_truth", Counter())
        reco_counter = info.get("particles_in_event_reco", Counter())

        tracklets = {
            t for pattern in ep.get_patterns()
            for vertex in pattern.get_vertices()
            for t in vertex.get_tracklets()
        }
        pattern_reco_counter = Counter(t.particle_id for t in tracklets)

        def to_named(counter):
            return Counter({
                particle_name_map.get(pid, particle_name_map["default"])["name"]: count
                for pid, count in counter.items()
            })

        pattern_named = frozenset(to_named(pattern_reco_counter).items())
        reco_named = frozenset(to_named(reco_counter).items())
        truth_named = frozenset(to_named(truth_counter).items())

        key = (pattern_named, reco_named, truth_named)
        error_groups[key] += 1

    # Build sizes and legend labels
    sizes = []
    legend_labels = []
    for (pattern_named, reco_named, truth_named), count in error_groups.items():
        sizes.append(count)
        label = (
            r"$\text{Pattern Particles: }" + generate_particle_label(Counter(dict(pattern_named))) + r"$" + "\n" +
            r"$\text{Tracklet Particles: }" + generate_particle_label(Counter(dict(reco_named))) + r"$" + "\n" +
            r"$\text{Truth Particles: }" + generate_particle_label(Counter(dict(truth_named))) + r"$"
        )
        legend_labels.append(label)

    # Sort
    sorted_indices = sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True)
    sizes = [sizes[i] for i in sorted_indices]
    legend_labels = [legend_labels[i] for i in sorted_indices]

    if len(legend_labels) > max_labels:
        other_size = sum(sizes[max_labels:])
        sizes = sizes[:max_labels] + [other_size]
        legend_labels = legend_labels[:max_labels] + [r"$\text{Other}$"]

    total = sum(sizes)

    fig, ax = plt.subplots(figsize=(10, 10))
    wedges, _, autotexts = ax.pie(
        sizes,
        labels=[f"Group {i+1}" for i in range(len(sizes))],
        autopct=lambda p: f'{p:.1f}% ({int(round(p * total / 100))})' if p > 1 else '',
        textprops={'fontsize': 9},
        startangle=140,
    )

    # Legend with LaTeX
    ax.legend(wedges, legend_labels, title="Event Types", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)

    # Title
    if title is None:
        title = f"Misclassified Events by Composition\n$N_{{\\text{{incorrect}}}} = {total}$"
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


