from __future__ import annotations

from .collection_profile import CollectionProfile
from .registry import register_collection_profile


@register_collection_profile
class TruthCollectionProfile(CollectionProfile):
    name = "truth"
    aliases = ()

    def required_fields(self) -> dict[str, str | None]:
        return {
            "patterns": "_Event_pattern_fitted_truth",
            "tracklets": "_Event_tracklets_truth",
            "fit_results": "_Event_fitres_truth",
            "hits": "_Event_atar_fr",
        }

    def optional_fields(self) -> dict[str, str | None]:
        return {
            "lyso_hits": "_Event_lyso_fr",
        }
