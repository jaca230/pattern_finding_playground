from __future__ import annotations

from figures.event_display.event_display_data import TimeCluster


class TimeClusterer:
    """Cluster event-display hits by gaps in observed hit time."""

    def __init__(
        self,
        gap_threshold_ns: float = 0.3,
        bridge_threshold_ns: float = 0.95,
        gap_fraction: float = 0.08,
        min_display_gap_ns: float = 0.15,
    ):
        self.gap_threshold_ns = gap_threshold_ns
        self.bridge_threshold_ns = bridge_threshold_ns
        self.gap_fraction = gap_fraction
        self.min_display_gap_ns = min_display_gap_ns

    def cluster(self, hits: list) -> list[TimeCluster]:
        timed_hits = [hit for hit in hits if hit.time is not None]
        timed_hits.sort(key=lambda hit: hit.time)
        if not timed_hits:
            return []

        groups = []
        current = [timed_hits[0]]
        for previous, current_hit, next_hit in self._with_neighbors(timed_hits):
            gap = abs(current_hit.time - previous.time)
            next_gap = abs(next_hit.time - current_hit.time) if next_hit is not None else None
            if self._is_cluster_break(gap, next_gap):
                groups.append(current)
                current = [current_hit]
            else:
                current.append(current_hit)
        groups.append(current)

        raw_widths = [max(hit.time for hit in group) - min(hit.time for hit in group) for group in groups]
        total_raw_width = sum(raw_widths)
        gap_size = max(total_raw_width * self.gap_fraction, self.min_display_gap_ns)

        clusters = []
        display_cursor = 0.0
        for index, group in enumerate(groups):
            raw_start = min(hit.time for hit in group)
            raw_stop = max(hit.time for hit in group)
            raw_width = raw_stop - raw_start
            display_start = display_cursor
            display_stop = display_cursor + raw_width
            clusters.append(
                TimeCluster(
                    index=index,
                    hits=tuple(group),
                    raw_start=raw_start,
                    raw_stop=raw_stop,
                    display_start=display_start,
                    display_stop=display_stop,
                )
            )
            display_cursor = display_stop + gap_size

        return clusters

    def display_time(self, hit, clusters: list[TimeCluster]) -> float | None:
        if hit.time is None:
            return None
        for cluster in clusters:
            if hit in cluster.hits:
                return cluster.display_start + (hit.time - cluster.raw_start)
        return hit.time

    def _with_neighbors(self, timed_hits):
        for index in range(1, len(timed_hits)):
            previous = timed_hits[index - 1]
            current = timed_hits[index]
            next_hit = timed_hits[index + 1] if index + 1 < len(timed_hits) else None
            yield previous, current, next_hit

    def _is_cluster_break(self, gap: float, next_gap: float | None) -> bool:
        if gap <= self.gap_threshold_ns:
            return False
        if gap >= self.bridge_threshold_ns:
            return True
        return next_gap is None or next_gap > self.gap_threshold_ns
