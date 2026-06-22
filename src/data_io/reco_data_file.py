from collections.abc import Mapping

import ROOT as r

from .cuts import CutFlow


class RecoDataFile:
    """Small owner for one reconstructed ROOT RNTuple and its geometry header."""

    REALISTIC_PREFIXES = {
        "patterns": "_Event_pattern_fitted_realistic",
        "tracklets": "_Event_tracklets",
        "fit_results": "_Event_fitres",
        "hits": "_Event_atar_fr",
    }
    TRUTH_PREFIXES = {
        "patterns": "_Event_pattern_fitted_truth",
        "tracklets": "_Event_tracklets_truth",
        "fit_results": "_Event_fitres_truth",
        "hits": "_Event_atar_fr",
    }
    _PREFIX_ALIASES = {
        "pattern": "patterns",
        "pattern_prefix": "patterns",
        "patterns_prefix": "patterns",
        "tracklet": "tracklets",
        "tracklet_prefix": "tracklets",
        "tracklets_prefix": "tracklets",
        "fit_result": "fit_results",
        "fit_result_prefix": "fit_results",
        "fit_results_prefix": "fit_results",
        "fitres": "fit_results",
        "hit": "hits",
        "hit_prefix": "hits",
        "hits_prefix": "hits",
    }

    def __init__(
        self,
        path: str,
        *,
        ntuple_name: str = "rec",
        prefix_map: Mapping[str, str | None] | None = None,
        truth_prefix_map: Mapping[str, str | None] | None = None,
        load_truth: bool = True,
        pattern_prefix: str | None = None,
        tracklet_prefix: str | None = None,
        fit_result_prefix: str | None = None,
        hit_prefix: str | None = None,
        cuts=None,
    ):
        self.path = path
        self.ntuple_name = ntuple_name
        self.prefixes = self._resolve_prefixes(
            prefix_map=prefix_map,
            pattern_prefix=pattern_prefix,
            tracklet_prefix=tracklet_prefix,
            fit_result_prefix=fit_result_prefix,
            hit_prefix=hit_prefix,
        )
        self.pattern_prefix = self.prefixes["patterns"]
        self.tracklet_prefix = self.prefixes["tracklets"]
        self.fit_result_prefix = self.prefixes["fit_results"]
        self.hit_prefix = self.prefixes["hits"]
        self.truth_prefixes = (
            self._resolve_truth_prefixes(truth_prefix_map)
            if load_truth
            else None
        )
        self.cut_flow = cuts if isinstance(cuts, CutFlow) else CutFlow(cuts)

        self.root_file = r.TFile(path, "READ")
        self.geo = self.root_file.Get("GeoHeader")
        if not self.geo:
            raise RuntimeError(f"No GeoHeader found in {path}")

        self._reader = r.ROOT.RNTupleReader.Open(ntuple_name, path)
        self._entry_obj = self._reader.CreateEntry()
        self._entries = int(self._reader.GetNEntries())
        self._pattern_token = self._entry_obj.GetToken(self.pattern_prefix)
        self._tracklet_token = self._entry_obj.GetToken(self.tracklet_prefix)
        self._fit_result_token = (
            self._entry_obj.GetToken(self.fit_result_prefix)
            if self.fit_result_prefix is not None
            else None
        )
        self._hit_token = self._entry_obj.GetToken(self.hit_prefix)
        self._truth_tokens = self._make_truth_tokens(self.truth_prefixes)

    def __len__(self) -> int:
        return self._entries

    @property
    def entries(self) -> int:
        return self._entries

    def load_entry(self, entry_index: int) -> dict:
        """Load one entry and return its raw ROOT collections.

        The returned collections are backed by ROOT's reusable RNTuple entry
        object. Use them immediately; a later call to `load_entry` will replace
        the loaded contents.
        """
        if entry_index < 0 or entry_index >= self._entries:
            raise IndexError(f"Entry {entry_index} outside dataset range [0, {self._entries})")

        self._reader.LoadEntry(entry_index, self._entry_obj)
        loaded = {
            "entry_index": entry_index,
            "patterns": self._entry_obj[self._pattern_token],
            "tracklets": self._entry_obj[self._tracklet_token],
            "hits": self._entry_obj[self._hit_token],
        }
        if self._fit_result_token is not None:
            loaded["fit_results"] = self._entry_obj[self._fit_result_token]
        if self.has_truth_entry:
            loaded["truth_entry"] = self._load_token_group(self._truth_tokens, entry_index)
        return loaded

    @property
    def has_truth_entry(self) -> bool:
        return self._truth_tokens is not None

    def add_cut(self, cut) -> None:
        self.cut_flow.add(cut)

    def clear_cuts(self) -> None:
        self.cut_flow.clear()

    def selected_entries(self, max_entries: int | None = None, progress=None) -> list[int]:
        return self.cut_flow.selected_entries(self, max_entries=max_entries, progress=progress)

    def _resolve_prefixes(
        self,
        *,
        prefix_map: Mapping[str, str | None] | None,
        pattern_prefix: str | None,
        tracklet_prefix: str | None,
        fit_result_prefix: str | None,
        hit_prefix: str | None,
    ) -> dict[str, str | None]:
        prefixes = dict(self.REALISTIC_PREFIXES)
        if prefix_map is not None:
            for key, value in prefix_map.items():
                prefixes[self._normalise_prefix_key(key)] = value

        explicit_prefixes = {
            "patterns": pattern_prefix,
            "tracklets": tracklet_prefix,
            "fit_results": fit_result_prefix,
            "hits": hit_prefix,
        }
        for key, value in explicit_prefixes.items():
            if value is not None:
                prefixes[key] = value

        missing = [key for key in ("patterns", "tracklets", "hits") if prefixes.get(key) is None]
        if missing:
            raise ValueError(f"Missing required RNTuple prefixes: {missing}")
        return prefixes

    def _resolve_truth_prefixes(self, truth_prefix_map: Mapping[str, str | None] | None) -> dict[str, str | None]:
        prefixes = dict(self.TRUTH_PREFIXES)
        if truth_prefix_map is not None:
            for key, value in truth_prefix_map.items():
                prefixes[self._normalise_prefix_key(key)] = value
        return prefixes

    def _make_truth_tokens(self, truth_prefixes: Mapping[str, str | None] | None) -> dict[str, object] | None:
        if truth_prefixes is None:
            return None

        tokens = {
            "patterns": self._optional_token(truth_prefixes["patterns"]),
            "tracklets": self._optional_token(truth_prefixes["tracklets"]),
            "hits": self._optional_token(truth_prefixes["hits"]),
            "fit_results": (
                self._optional_token(truth_prefixes["fit_results"])
                if truth_prefixes.get("fit_results") is not None
                else None
            ),
        }
        if any(tokens[key] is None for key in ("patterns", "tracklets", "hits")):
            return None
        return tokens

    def _optional_token(self, field_name: str):
        try:
            return self._entry_obj.GetToken(field_name)
        except Exception:
            return None

    def _load_token_group(self, tokens: Mapping[str, object], entry_index: int) -> dict:
        loaded = {
            "entry_index": entry_index,
            "patterns": self._entry_obj[tokens["patterns"]],
            "tracklets": self._entry_obj[tokens["tracklets"]],
            "hits": self._entry_obj[tokens["hits"]],
        }
        if tokens.get("fit_results") is not None:
            loaded["fit_results"] = self._entry_obj[tokens["fit_results"]]
        return loaded

    def _normalise_prefix_key(self, key: str) -> str:
        normalised = self._PREFIX_ALIASES.get(key, key)
        if normalised not in self.REALISTIC_PREFIXES:
            candidates = ", ".join(sorted(self.REALISTIC_PREFIXES))
            raise ValueError(f"Unknown prefix key '{key}'. Candidates are: {candidates}")
        return normalised
