from __future__ import annotations

from collections.abc import Mapping

from .collection_profile import CollectionProfile
from .inline_collection_profile import InlineCollectionProfile


_FIELD_ALIASES: dict[str, str] = {
    "pattern": "patterns",
    "tracklet": "tracklets",
    "fit_result": "fit_results",
    "fitres": "fit_results",
    "hit": "hits",
    "patterns_field": "patterns",
    "tracklets_field": "tracklets",
    "fit_results_field": "fit_results",
    "hits_field": "hits",
}


class CollectionProfileRegistry:
    def __init__(self):
        self._registry: dict[str, type[CollectionProfile]] = {}
        self._canonical_names: dict[str, type[CollectionProfile]] = {}

    def register(self, profile_cls: type[CollectionProfile]) -> None:
        name = getattr(profile_cls, "name", None)
        if not name:
            raise ValueError(f"Collection profile class {profile_cls.__name__} must declare a canonical name.")
        if name in self._registry:
            raise ValueError(f"Collection profile name '{name}' is already registered.")

        self._registry[name] = profile_cls
        self._canonical_names[name] = profile_cls

        aliases = getattr(profile_cls, "aliases", ())
        for alias in aliases:
            if alias in self._registry:
                raise ValueError(f"Collection profile alias '{alias}' is already registered.")
            self._registry[alias] = profile_cls

    def available_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._canonical_names))

    def create(self, profile: str | Mapping[str, str | None] | CollectionProfile) -> CollectionProfile:
        if isinstance(profile, CollectionProfile):
            return profile
        if isinstance(profile, str):
            profile_cls = self._registry.get(profile)
            if profile_cls is None:
                candidates = ", ".join(sorted(self._registry))
                raise ValueError(f"Unknown collection profile '{profile}'. Candidates are: {candidates}")
            return profile_cls()

        fields = {
            normalise_collection_field_name(key): value
            for key, value in profile.items()
        }
        return InlineCollectionProfile(fields)


REGISTRY = CollectionProfileRegistry()


def register_collection_profile(profile_cls: type[CollectionProfile]) -> type[CollectionProfile]:
    REGISTRY.register(profile_cls)
    return profile_cls


def available_profile_names() -> tuple[str, ...]:
    return REGISTRY.available_names()


def normalise_collection_field_name(name: str) -> str:
    normalised = _FIELD_ALIASES.get(name, name)
    valid_fields = {"patterns", "tracklets", "fit_results", "hits", "lyso_hits"}
    if normalised not in valid_fields:
        candidates = ", ".join(sorted(valid_fields))
        raise ValueError(f"Unknown collection field '{name}'. Candidates are: {candidates}")
    return normalised


def create_collection_profile(
    profile: str | Mapping[str, str | None] | CollectionProfile,
) -> CollectionProfile:
    return REGISTRY.create(profile)
