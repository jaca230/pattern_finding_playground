from __future__ import annotations

from .collection_profile import CollectionProfile


class InlineCollectionProfile(CollectionProfile):
    """Ad hoc collection profile built directly from field names."""

    def __init__(self, fields: dict[str, str | None]):
        self._fields = dict(fields)

    def required_fields(self) -> dict[str, str | None]:
        return {
            "patterns": self._fields.get("patterns"),
            "tracklets": self._fields.get("tracklets"),
            "fit_results": self._fields.get("fit_results"),
            "hits": self._fields.get("hits"),
        }

    def optional_fields(self) -> dict[str, str | None]:
        return {
            key: value
            for key, value in self._fields.items()
            if key not in {"patterns", "tracklets", "fit_results", "hits"}
        }
