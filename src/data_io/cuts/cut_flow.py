from collections.abc import Iterable

from .data_cut import DataCut


class CutFlow:
    def __init__(self, cuts: Iterable[DataCut] | None = None):
        self.cuts = list(cuts or [])
        self.last_report: dict | None = None

    def add(self, cut: DataCut) -> None:
        self.cuts.append(cut)

    def clear(self) -> None:
        self.cuts.clear()
        self.last_report = None

    def enabled_cuts(self) -> list[DataCut]:
        return [cut for cut in self.cuts if cut.enabled]

    def accepts(self, data_file, entry: dict) -> bool:
        return all(cut(data_file, entry) for cut in self.cuts)

    def selected_entries(self, data_file, max_entries: int | None = None, progress=None) -> list[int]:
        entries = data_file.entries if max_entries is None else min(data_file.entries, max_entries)
        entry_range = range(entries)
        if progress is not None:
            entry_range = progress(entry_range, desc="Applying dataset cuts")

        selected = []
        enabled_cuts = self.enabled_cuts()
        stage_removed = [0 for _ in enabled_cuts]
        cumulative_removed = 0
        for entry_index in entry_range:
            entry = data_file.load_entry(entry_index)
            accepted = True
            for cut_index, cut in enumerate(enabled_cuts):
                if cut(data_file, entry):
                    continue
                stage_removed[cut_index] += 1
                cumulative_removed += 1
                accepted = False
                break
            if accepted:
                selected.append(entry_index)

        self.last_report = self._build_report(entries, enabled_cuts, stage_removed, len(selected), cumulative_removed)
        return selected

    def format_last_report(self) -> str:
        if self.last_report is None:
            return "No cut-flow report is available yet. Run selected_entries(...) first."

        total_scanned = self.last_report["total_scanned"]
        rows = self.last_report["rows"]
        lines = []
        header = (
            f"{'cut':<24} {'removed':>10} {'removed %':>10} "
            f"{'cumulative':>12} {'cumulative %':>14}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for row in rows:
            lines.append(
                f"{row['cut']:<24} "
                f"{row['removed']:>10d} "
                f"{row['removed_pct_total']:>9.2f}% "
                f"{row['cumulative_removed']:>12d} "
                f"{row['cumulative_removed_pct_total']:>13.2f}%"
            )

        totals = self.last_report["totals"]
        lines.append("-" * len(header))
        lines.append(
            f"{'TOTAL':<24} "
            f"{totals['total_removed']:>10d} "
            f"{totals['total_removed_pct']:>9.2f}% "
            f"{totals['total_removed']:>12d} "
            f"{totals['total_removed_pct']:>13.2f}%"
        )
        lines.append(
            f"{'KEPT':<24} "
            f"{totals['total_kept']:>10d} "
            f"{totals['total_kept_pct']:>9.2f}% "
            f"{totals['total_kept']:>12d} "
            f"{totals['total_kept_pct']:>13.2f}%"
        )
        lines.append(f"Scanned entries: {total_scanned}")
        return "\n".join(lines)

    def print_last_report(self) -> None:
        print(self.format_last_report())

    def _build_report(
        self,
        total_scanned: int,
        enabled_cuts: list[DataCut],
        stage_removed: list[int],
        total_kept: int,
        total_removed: int,
    ) -> dict:
        rows = []
        cumulative_removed = 0
        for cut, removed in zip(enabled_cuts, stage_removed):
            cumulative_removed += removed
            rows.append(
                {
                    "cut": cut.name,
                    "removed": removed,
                    "removed_pct_total": self._pct(removed, total_scanned),
                    "cumulative_removed": cumulative_removed,
                    "cumulative_removed_pct_total": self._pct(cumulative_removed, total_scanned),
                }
            )

        return {
            "total_scanned": total_scanned,
            "rows": rows,
            "totals": {
                "total_removed": total_removed,
                "total_removed_pct": self._pct(total_removed, total_scanned),
                "total_kept": total_kept,
                "total_kept_pct": self._pct(total_kept, total_scanned),
            },
        }

    def _pct(self, value: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return 100.0 * float(value) / float(total)
