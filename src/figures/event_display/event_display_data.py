from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Any

from figures.event_display.geometry.strip_rectangle import StripRectangle


@dataclass(frozen=True)
class HitGlyph:
    hit: Any
    tracklet: Any | None
    plane: str
    rectangle: StripRectangle
    energy: float
    time: float | None
    particle_id: int


@dataclass(frozen=True)
class AggregatedStripGlyph:
    plane: str
    rectangle: StripRectangle
    particle_counts: Counter
    hit_count: int
    total_energy: float


@dataclass(frozen=True)
class TimeCluster:
    index: int
    hits: tuple[Any, ...]
    raw_start: float
    raw_stop: float
    display_start: float
    display_stop: float

    @property
    def raw_width(self) -> float:
        return self.raw_stop - self.raw_start
