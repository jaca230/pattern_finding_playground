"""Small compatibility layer for current Reco RNTuple outputs.

The original notebooks in this project were written against an older ROOT
TTree layout where `tree.GetEntry(i)` populated `tree.patternVec` and
`tree.trackletVec`. Current Reco outputs store the same logical collections in
an RNTuple. This module exposes only the tiny old-style surface used by the
notebooks and algorithms in this playground.
"""

from __future__ import annotations

from typing import Sequence

import ROOT as r


DEFAULT_LIBRARIES = (
    "libpi_utils.so",
    "libpi_headers.so",
    "libpi_MonteCarlo.so",
    "libpi_Reco.so",
    "libPiGaudiData.so",
)


def load_pioneer_libraries(lib_dir: str = "/simulation/docker/install/lib") -> None:
    """Load the split PIONEER ROOT dictionaries used by current builds."""
    for library in DEFAULT_LIBRARIES:
        r.gSystem.Load(f"{lib_dir}/{library}")


class GeoHeaderHandle:
    """Keep the ROOT file alive while delegating calls to its GeoHeader."""

    def __init__(self, path: str):
        self.file = r.TFile(path, "READ")
        self.header = self.file.Get("GeoHeader")
        if not self.header:
            raise RuntimeError(f"No GeoHeader found in {path}")

    def __getattr__(self, name: str):
        return getattr(self.header, name)


def open_geo_header(path: str) -> GeoHeaderHandle:
    return GeoHeaderHandle(path)


class TrackletProxy:
    """Add the old notebook's `GetAllHits()` method to current tracklets."""

    def __init__(self, tracklet, hit_vector):
        self.tracklet = tracklet
        self.hit_vector = hit_vector

    def GetPID(self) -> int:
        return int(self.tracklet.GetPID())

    def GetEID(self) -> int:
        return int(self.tracklet.GetEID())

    def GetAllHits(self) -> Sequence[object]:
        return [
            self.hit_vector[int(index)]
            for index in self.tracklet.GetAtarHitIndices()
            if 0 <= int(index) < self.hit_vector.size()
        ]


class RNTupleRecoTree:
    """Expose minimal old-TTree-like access to current Reco RNTuples."""

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

        self.patternVec = []
        self.trackletVec: list[TrackletProxy] = []
        self.hitVec = []
        self.infoVec: list[object] = []
        self._entry = -1

    def GetEntries(self) -> int:
        return self._entries

    def GetEntry(self, entry: int) -> int:
        if entry < 0 or entry >= self._entries:
            return 0

        self._reader.LoadEntry(entry, self._entry_obj)
        self.patternVec = self._entry_obj[self._pattern_token]
        self.hitVec = self._entry_obj[self._hit_token]
        raw_tracklets = self._entry_obj[self._tracklet_token]
        self.trackletVec = [
            TrackletProxy(raw_tracklets[index], self.hitVec)
            for index in range(raw_tracklets.size())
        ]
        self.infoVec = []
        self._entry = entry
        return 1


def find_first_event_with_patterns(tree: RNTupleRecoTree, max_events: int | None = None) -> int:
    """Return the first entry with at least one pattern and one tracklet."""
    entries = tree.GetEntries() if max_events is None else min(tree.GetEntries(), max_events)
    for entry in range(entries):
        tree.GetEntry(entry)
        if tree.patternVec and tree.trackletVec:
            return entry
    raise RuntimeError("No event with reconstructed patterns found.")
