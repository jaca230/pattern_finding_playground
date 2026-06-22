from __future__ import annotations

import math
from typing import Any

from figures.event_display.geometry.strip_rectangle import StripRectangle


class AtarGeometryProjector:
    """Project ATAR strip volumes into 2D display rectangles."""

    def __init__(
        self,
        geo: Any | None,
        *,
        fallback_z_half_width: float = 0.06,
        fallback_coord_half_width: float = 0.10,
    ):
        self.geo = geo
        self.fallback_z_half_width = fallback_z_half_width
        self.fallback_coord_half_width = fallback_coord_half_width
        self._cache: dict[tuple[int, str], StripRectangle] = {}

    def rectangle_for_hit(self, hit: Any, plane: str) -> StripRectangle:
        if hit.volume_id is not None and self.geo is not None:
            cache_key = (int(hit.volume_id), plane)
            if cache_key not in self._cache:
                self._cache[cache_key] = self._rectangle_from_geometry(int(hit.volume_id), hit, plane)
            return self._cache[cache_key]

        return self._fallback_rectangle(hit, plane)

    def guide_rectangles(self, glyphs: list[Any], plane: str) -> list[StripRectangle]:
        seen = set()
        guides = []
        for glyph in glyphs:
            rect = glyph.rectangle
            key = (round(rect.z_min, 5), round(rect.z_max, 5))
            if key not in seen:
                seen.add(key)
                guides.append(rect)
        return sorted(guides, key=lambda rect: rect.z_center)

    def _rectangle_from_geometry(self, volume_id: int, hit: Any, plane: str) -> StripRectangle:
        try:
            geometry_type = self.geo.GetGeometryType(volume_id).Data()
            if geometry_type != "G4Box":
                return self._fallback_rectangle(hit, plane)

            hx = float(self.geo.GetBoxHalfLengthX(volume_id))
            hy = float(self.geo.GetBoxHalfLengthY(volume_id))
            hz = float(self.geo.GetBoxHalfLengthZ(volume_id))
            points = self._box_global_corners(volume_id, hx, hy, hz)
            zs = [point[2] for point in points]
            if plane == "xz":
                coords = [point[0] for point in points]
            elif plane == "yz":
                coords = [point[1] for point in points]
            else:
                raise ValueError(f"Unknown ATAR display plane: {plane}")

            return StripRectangle(
                z_min=min(zs),
                z_max=max(zs),
                coord_min=min(coords),
                coord_max=max(coords),
            )
        except Exception:
            return self._fallback_rectangle(hit, plane)

    def _box_global_corners(self, volume_id: int, hx: float, hy: float, hz: float) -> list[tuple[float, float, float]]:
        import ROOT as r

        points = []
        for sx in (-hx, hx):
            for sy in (-hy, hy):
                for sz in (-hz, hz):
                    point = self.geo.LocalToGlobal(volume_id, r.ROOT.Math.XYZPoint(sx, sy, sz))
                    points.append((float(point.X()), float(point.Y()), float(point.Z())))
        return points

    def _fallback_rectangle(self, hit: Any, plane: str) -> StripRectangle:
        coord = hit.x if plane == "xz" else hit.y
        if coord is None or not math.isfinite(float(coord)):
            coord = 0.0
        z = 0.0 if hit.z is None else float(hit.z)

        return StripRectangle(
            z_min=z - self.fallback_z_half_width,
            z_max=z + self.fallback_z_half_width,
            coord_min=float(coord) - self.fallback_coord_half_width,
            coord_max=float(coord) + self.fallback_coord_half_width,
        )
