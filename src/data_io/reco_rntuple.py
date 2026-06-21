from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import ROOT as r


@dataclass
class RecoTracklet:
    raw: object
    hit_vector: object

    @property
    def pid(self) -> int:
        return int(self.raw.GetPID())

    @property
    def eid(self) -> int:
        return int(self.raw.GetEID())

    @property
    def hits(self) -> Sequence[object]:
        return [
            self.hit_vector[int(index)]
            for index in self.raw.GetAtarHitIndices()
            if 0 <= int(index) < self.hit_vector.size()
        ]

    @property
    def start_point(self):
        return self.raw.GetStartPoint()

    @property
    def stop_point(self):
        return self.raw.GetStopPoint()


@dataclass
class RecoEvent:
    entry_index: int
    patterns: object
    tracklets: list[RecoTracklet]
    hits: object

    @property
    def has_patterns(self) -> bool:
        return bool(self.patterns) and bool(self.tracklets)


class RecoRNTupleDataset:
    def __init__(
        self,
        path: str,
        *,
        ntuple_name: str = "rec",
        pattern_prefix: str,
        tracklet_prefix: str,
        hit_prefix: str = "_Event_atar_fr",
    ):
        self.path = path
        self.ntuple_name = ntuple_name
        self.pattern_prefix = pattern_prefix
        self.tracklet_prefix = tracklet_prefix
        self.hit_prefix = hit_prefix

        self._reader = r.ROOT.RNTupleReader.Open(ntuple_name, path)
        self._entry_obj = self._reader.CreateEntry()
        self._entries = int(self._reader.GetNEntries())
        self._pattern_token = self._entry_obj.GetToken(pattern_prefix)
        self._tracklet_token = self._entry_obj.GetToken(tracklet_prefix)
        self._hit_token = self._entry_obj.GetToken(hit_prefix)

    def __len__(self) -> int:
        return self._entries

    @property
    def entries(self) -> int:
        return self._entries

    def get_event(self, entry_index: int) -> RecoEvent:
        if entry_index < 0 or entry_index >= self._entries:
            raise IndexError(f"Entry {entry_index} outside dataset range [0, {self._entries})")

        self._reader.LoadEntry(entry_index, self._entry_obj)
        patterns = self._entry_obj[self._pattern_token]
        hits = self._entry_obj[self._hit_token]
        raw_tracklets = self._entry_obj[self._tracklet_token]
        tracklets = [
            RecoTracklet(raw_tracklets[index], hits)
            for index in range(raw_tracklets.size())
        ]

        return RecoEvent(
            entry_index=entry_index,
            patterns=patterns,
            tracklets=tracklets,
            hits=hits,
        )


def find_first_event_with_patterns(dataset: RecoRNTupleDataset, max_events: int | None = None) -> int:
    entries = dataset.entries if max_events is None else min(dataset.entries, max_events)
    for entry_index in range(entries):
        if dataset.get_event(entry_index).has_patterns:
            return entry_index
    raise RuntimeError("No event with reconstructed patterns found.")
