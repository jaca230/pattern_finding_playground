from __future__ import annotations

from .collection_profile import CollectionProfile
from .registry import register_collection_profile


@register_collection_profile
class RealisticCollectionProfile(CollectionProfile):
    name = "realistic"
    aliases = ("reco",)

    def required_fields(self) -> dict[str, str | None]:
        return {
            "patterns": "_Event_pattern_fitted_realistic",
            "tracklets": "_Event_tracklets",
            "fit_results": "_Event_fitres",
            "hits": "_Event_atar_fr",
        }

    def optional_fields(self) -> dict[str, str | None]:
        return {
            "lyso_hits": "_Event_lyso_fr",
        }
