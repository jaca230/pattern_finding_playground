from .registry import register_cut
from .data_cut import DataCut


@register_cut(
    name="delayed_lyso_cluster",
    description="Require one delayed LYSO activity cluster above threshold within a time window.",
    parameters={
        "min_cluster_energy_mev": "Minimum summed observed cluster energy in MeV.",
        "window_start_ns": "Start of the accepted hit-time window in ns.",
        "window_stop_ns": "End of the accepted hit-time window in ns.",
        "cluster_gap_ns": "Maximum time gap in ns for hits to stay in the same cluster.",
        "enabled": "Set False to keep the cut configured but inactive.",
    },
    example=(
        "DelayedLysoClusterCut(min_cluster_energy_mev=10.0, "
        "window_start_ns=0.0, window_stop_ns=700.0, cluster_gap_ns=5.0, enabled=True)"
    ),
)
class DelayedLysoClusterCut(DataCut):
    """Require one delayed LYSO activity cluster above threshold in a time window.

    The cut looks at reconstructed fast-response LYSO hits, keeps only hits inside
    the requested time window, clusters them by a simple maximum time gap, and
    requires at least one cluster whose summed observed energy exceeds the
    configured threshold.
    """

    def __init__(
        self,
        *,
        min_cluster_energy_mev: float = 10.0,
        window_start_ns: float = 0.0,
        window_stop_ns: float = 700.0,
        cluster_gap_ns: float = 5.0,
        enabled: bool = True,
    ):
        super().__init__(name="delayed_lyso_cluster", enabled=enabled)
        self.min_cluster_energy_mev = float(min_cluster_energy_mev)
        self.window_start_ns = float(window_start_ns)
        self.window_stop_ns = float(window_stop_ns)
        self.cluster_gap_ns = float(cluster_gap_ns)

    def accepts(self, data_file, entry: dict) -> bool:
        lyso_hits = entry.get("lyso_hits")
        if lyso_hits is None:
            return False

        timed_hits = []
        for hit in lyso_hits:
            time_ns = float(hit.GetObservedTime())
            if time_ns < self.window_start_ns or time_ns > self.window_stop_ns:
                continue
            timed_hits.append((time_ns, float(hit.GetObservedEdep())))

        if not timed_hits:
            return False

        timed_hits.sort(key=lambda item: item[0])
        cluster_energy = timed_hits[0][1]
        previous_time = timed_hits[0][0]

        for time_ns, energy_mev in timed_hits[1:]:
            if time_ns - previous_time <= self.cluster_gap_ns:
                cluster_energy += energy_mev
            else:
                if cluster_energy >= self.min_cluster_energy_mev:
                    return True
                cluster_energy = energy_mev
            previous_time = time_ns

        return cluster_energy >= self.min_cluster_energy_mev
