from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StripRectangle:
    """Projected ATAR strip footprint in one display plane."""

    z_min: float
    z_max: float
    coord_min: float
    coord_max: float

    @property
    def z_center(self) -> float:
        return 0.5 * (self.z_min + self.z_max)

    @property
    def coord_center(self) -> float:
        return 0.5 * (self.coord_min + self.coord_max)

    @property
    def z_width(self) -> float:
        return self.z_max - self.z_min

    @property
    def coord_width(self) -> float:
        return self.coord_max - self.coord_min
