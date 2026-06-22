from collections.abc import Iterable

from .data_cut import DataCut


class CutFlow:
    def __init__(self, cuts: Iterable[DataCut] | None = None):
        self.cuts = list(cuts or [])

    def add(self, cut: DataCut) -> None:
        self.cuts.append(cut)

    def clear(self) -> None:
        self.cuts.clear()

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
        for entry_index in entry_range:
            entry = data_file.load_entry(entry_index)
            if self.accepts(data_file, entry):
                selected.append(entry_index)
        return selected
