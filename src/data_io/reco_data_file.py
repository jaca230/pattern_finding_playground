from __future__ import annotations

from collections.abc import Mapping

import ROOT as r

from .cuts import CutFlow
from .profiles import (
    CollectionProfile,
    available_profile_names,
    create_collection_profile,
    normalise_collection_field_name,
)


class RecoDataFile:
    """Small owner for one reconstructed ROOT RNTuple and its geometry header."""

    def __init__(
        self,
        path: str,
        *,
        ntuple_name: str = "rec",
        profile: str | Mapping[str, str | None] | CollectionProfile = "realistic",
        truth_profile: str | Mapping[str, str | None] | CollectionProfile = "truth",
        load_truth: bool = True,
        field_overrides: Mapping[str, str | None] | None = None,
        truth_field_overrides: Mapping[str, str | None] | None = None,
        lyso_hit_field: str | None = None,
        cuts=None,
    ):
        self.path = path
        self.ntuple_name = ntuple_name
        self.profile = create_collection_profile(profile)
        self.fields = self._resolved_fields(self.profile, field_overrides, label="profile")
        self.pattern_field = self.fields["patterns"]
        self.tracklet_field = self.fields["tracklets"]
        self.fit_result_field = self.fields["fit_results"]
        self.hit_field = self.fields["hits"]
        self.truth_fields = (
            self._resolved_fields(
                create_collection_profile(truth_profile),
                truth_field_overrides,
                label="truth_profile",
            )
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
        self._pattern_token = self._entry_obj.GetToken(self.pattern_field)
        self._tracklet_token = self._entry_obj.GetToken(self.tracklet_field)
        self._fit_result_token = (
            self._entry_obj.GetToken(self.fit_result_field)
            if self.fit_result_field is not None
            else None
        )
        self._hit_token = self._entry_obj.GetToken(self.hit_field)
        resolved_lyso_field = (
            lyso_hit_field
            if lyso_hit_field is not None
            else self.profile.optional_fields().get("lyso_hits")
        )
        self._lyso_hit_token = self._optional_token(resolved_lyso_field)
        self._truth_tokens = self._make_truth_tokens(self.truth_fields)

    def __len__(self) -> int:
        return self._entries

    @property
    def entries(self) -> int:
        return self._entries

    @classmethod
    def available_profiles(cls) -> tuple[str, ...]:
        return available_profile_names()

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
        if self._lyso_hit_token is not None:
            loaded["lyso_hits"] = self._entry_obj[self._lyso_hit_token]
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

    def format_cut_table(self) -> str:
        return self.cut_flow.format_last_report()

    def print_cut_table(self) -> None:
        self.cut_flow.print_last_report()

    def _resolved_fields(
        self,
        profile: CollectionProfile,
        overrides: Mapping[str, str | None] | None,
        *,
        label: str,
    ) -> dict[str, str | None]:
        normalised_overrides = None
        if overrides is not None:
            normalised_overrides = {
                normalise_collection_field_name(key): value
                for key, value in overrides.items()
            }
        return profile.resolve_fields(normalised_overrides, label=label)

    def _make_truth_tokens(self, truth_fields: Mapping[str, str | None] | None) -> dict[str, object] | None:
        if truth_fields is None:
            return None

        tokens = {
            "patterns": self._optional_token(truth_fields["patterns"]),
            "tracklets": self._optional_token(truth_fields["tracklets"]),
            "hits": self._optional_token(truth_fields["hits"]),
            "fit_results": (
                self._optional_token(truth_fields["fit_results"])
                if truth_fields.get("fit_results") is not None
                else None
            ),
        }
        if any(tokens[key] is None for key in ("patterns", "tracklets", "hits")):
            return None
        return tokens

    def _optional_token(self, field_name: str | None):
        if field_name is None:
            return None
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
