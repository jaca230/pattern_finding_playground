from __future__ import annotations

from abc import ABC


class CollectionProfile(ABC):
    """Base class for named ROOT collection layouts used by RecoDataFile."""

    aliases: tuple[str, ...] = ()

    def required_fields(self) -> dict[str, str | None]:
        raise NotImplementedError

    def optional_fields(self) -> dict[str, str | None]:
        return {}

    def all_fields(self) -> dict[str, str | None]:
        fields = dict(self.required_fields())
        fields.update(self.optional_fields())
        return fields

    def resolve_fields(
        self,
        overrides: dict[str, str | None] | None = None,
        *,
        label: str = "profile",
    ) -> dict[str, str | None]:
        fields = dict(self.required_fields())
        if overrides is not None:
            fields.update(overrides)

        missing = [key for key in ("patterns", "tracklets", "hits") if fields.get(key) is None]
        if missing:
            raise ValueError(f"{label} is missing required collection fields: {missing}")

        return fields
