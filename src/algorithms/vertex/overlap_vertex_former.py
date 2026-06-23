from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import math

from algorithms.registry import register_algorithm
from algorithms.vertex.vertex_former import VertexFormer
from models.tracklet import Tracklet
from models.vertex import Vertex


@dataclass(frozen=True)
class _FitEndpoint:
    tracklet_id: int
    endpoint_index: int
    position: tuple[float, float, float]
    linear_heading: tuple[float, float, float]


@register_algorithm(
    "vertex",
    name="overlap",
    description="Match fitted tracklet endpoints by projected overlap, with a fallback closest-approach repair.",
)
class OverlapVertexFormer(VertexFormer):
    """Python analogue of Reco's PFOverlapVertexFormer.

    This follows Sean's current C++ default vertexing rule: compare fitted
    tracklet endpoints in the x-z and y-z ATAR projections, allow one
    extrapolated closest-approach repair if only one projection overlaps, then
    build event patterns from shared-tracklet connected components downstream.
    """

    ELLIPSE_OVERLAP = "ellipse_overlap"
    EXTRAP_DCA_OVERLAP = "extrap_dca_overlap"
    ORIENTATION_MISSING = "orientation_missing"
    NO_OVERLAP = "no_overlap"
    SINGLE_ENDPOINT = "single_endpoint"

    PION_PID = 211
    POSITRON_PID = -11
    MUON_PID = -13
    ENDPOINT_Z_MARGIN_MM = 1.0
    ENDPOINT_COORD_MARGIN_MM = 1.0

    def __init__(self, distance_threshold: float = 1.0):
        self.distance_threshold = float(distance_threshold)

    def form_vertices(
        self,
        tracklets: set[Tracklet],
        storage: Optional[Any] = None,
    ) -> tuple[set[Vertex], dict]:
        vertices: set[Vertex] = set()
        vertex_id = 0

        sorted_tracklets = sorted(tracklets, key=lambda tracklet: tracklet.tracklet_id)
        pion_tracklets = [t for t in sorted_tracklets if t.particle_id == self.PION_PID]
        positron_tracklets = [t for t in sorted_tracklets if t.particle_id == self.POSITRON_PID]
        muon_tracklets = [t for t in sorted_tracklets if t.particle_id == self.MUON_PID]

        tracklet_from_endpoint: dict[_FitEndpoint, Tracklet] = {}
        all_endpoints: set[_FitEndpoint] = set()
        for tracklet in sorted_tracklets:
            for endpoint in self._fit_endpoints(tracklet):
                tracklet_from_endpoint[endpoint] = tracklet
                all_endpoints.add(endpoint)

        for pion_tracklet in pion_tracklets:
            candidates = positron_tracklets + muon_tracklets
            vertex = self._determine_best_vertex(vertex_id, pion_tracklet, candidates)
            if self._is_valid(vertex):
                vertices.add(vertex)
                vertex_id += 1
                self._remove_vertex_endpoints(all_endpoints, vertex)

        for muon_tracklet in muon_tracklets:
            vertex = self._determine_best_vertex(vertex_id, muon_tracklet, positron_tracklets)
            if self._is_valid(vertex):
                vertices.add(vertex)
                vertex_id += 1
                self._remove_vertex_endpoints(all_endpoints, vertex)

        for endpoint in sorted(all_endpoints, key=lambda ep: (ep.tracklet_id, ep.endpoint_index)):
            vertex = self._single_endpoint_vertex(
                vertex_id,
                tracklet_from_endpoint[endpoint],
                endpoint,
            )
            vertices.add(vertex)
            vertex_id += 1

        result_info = {
            "distance_threshold": self.distance_threshold,
            "vertices_count": len(vertices),
            "input_tracklets_count": len(tracklets),
            "single_endpoint_vertices": sum(
                1 for vertex in vertices if vertex.extra_info.get("front_overlap_type") == self.SINGLE_ENDPOINT
            ),
        }

        return vertices, result_info

    def _determine_best_vertex(
        self,
        vertex_id: int,
        seed_tracklet: Tracklet,
        candidate_tracklets: list[Tracklet],
    ) -> Vertex:
        best_vertex = self._invalid_vertex(vertex_id)
        for candidate_tracklet in candidate_tracklets:
            candidate_vertex = self._determine_candidate_vertex(
                vertex_id,
                seed_tracklet,
                candidate_tracklet,
            )
            if not self._is_valid(candidate_vertex):
                continue
            if not self._is_valid(best_vertex) or self._score(candidate_vertex) < self._score(best_vertex):
                best_vertex = candidate_vertex
        return best_vertex

    def _determine_candidate_vertex(
        self,
        vertex_id: int,
        tracklet_1: Tracklet,
        tracklet_2: Tracklet,
    ) -> Vertex:
        vertex = self._invalid_vertex(vertex_id)
        vertex.add_tracklet(tracklet_1)
        vertex.add_tracklet(tracklet_2)

        overlap_xz = self._determine_overlap(tracklet_1, tracklet_2, "xz")
        overlap_yz = self._determine_overlap(tracklet_1, tracklet_2, "yz")

        vertex.extra_info.update(
            {
                "front_overlap_type": overlap_xz["type"],
                "back_overlap_type": overlap_yz["type"],
                "front_distance": overlap_xz["distance"],
                "back_distance": overlap_yz["distance"],
                "tracklets_time_ordered": self._tracklet_time(tracklet_1) <= self._tracklet_time(tracklet_2),
            }
        )
        self._set_overlap_endpoint_info(vertex, "front", overlap_xz)
        self._set_overlap_endpoint_info(vertex, "back", overlap_yz)

        if overlap_xz["type"] == self.ELLIPSE_OVERLAP and overlap_yz["type"] == self.ELLIPSE_OVERLAP:
            vertex.extra_info["valid"] = True
        elif overlap_xz["type"] == self.ELLIPSE_OVERLAP and overlap_yz["type"] == self.ORIENTATION_MISSING:
            vertex.extra_info["valid"] = True
        elif overlap_xz["type"] == self.ORIENTATION_MISSING and overlap_yz["type"] == self.ELLIPSE_OVERLAP:
            vertex.extra_info["valid"] = True
        elif overlap_xz["type"] == self.ELLIPSE_OVERLAP and overlap_yz["type"] == self.NO_OVERLAP:
            self._try_extrapolated_overlap(vertex, overlap_yz, "yz", "back")
        elif overlap_xz["type"] == self.NO_OVERLAP and overlap_yz["type"] == self.ELLIPSE_OVERLAP:
            self._try_extrapolated_overlap(vertex, overlap_xz, "xz", "front")

        return vertex

    def _determine_overlap(self, tracklet_1: Tracklet, tracklet_2: Tracklet, orientation: str) -> dict[str, Any]:
        endpoints_1 = self._fit_endpoints(tracklet_1, orientation)
        endpoints_2 = self._fit_endpoints(tracklet_2, orientation)
        min_distance = math.inf
        closest_1 = None
        closest_2 = None

        for endpoint_1 in endpoints_1:
            for endpoint_2 in endpoints_2:
                distance = self._projected_distance(endpoint_1, endpoint_2, orientation)
                if distance < min_distance:
                    min_distance = distance
                    closest_1 = endpoint_1
                    closest_2 = endpoint_2

        if min_distance < self.distance_threshold:
            overlap_type = self.ELLIPSE_OVERLAP
        elif closest_1 is None or closest_2 is None:
            overlap_type = self.ORIENTATION_MISSING
        else:
            overlap_type = self.NO_OVERLAP

        return {
            "type": overlap_type,
            "distance": min_distance,
            "endpoint_1": closest_1,
            "endpoint_2": closest_2,
        }

    def _try_extrapolated_overlap(
        self,
        vertex: Vertex,
        overlap: dict[str, Any],
        orientation: str,
        vertex_side: str,
    ) -> None:
        endpoint_1 = overlap["endpoint_1"]
        endpoint_2 = overlap["endpoint_2"]
        if endpoint_1 is None or endpoint_2 is None:
            return

        dca, extrapolated_position = self._compute_extrapolated_dca(endpoint_2, endpoint_1, orientation)
        if dca < self.distance_threshold:
            vertex.extra_info["valid"] = True
            vertex.extra_info[f"{vertex_side}_overlap_type"] = self.EXTRAP_DCA_OVERLAP
            vertex.extra_info[f"{vertex_side}_distance"] = dca
            vertex.extra_info[f"{vertex_side}_vertex_position"] = self._midpoint(
                extrapolated_position,
                endpoint_1.position,
            )

    def _compute_extrapolated_dca(
        self,
        endpoint: _FitEndpoint,
        endpoint_target: _FitEndpoint,
        orientation: str,
    ) -> tuple[float, tuple[float, float, float]]:
        heading = list(endpoint.linear_heading)
        if orientation == "xz":
            heading[1] = 0.0
        elif orientation == "yz":
            heading[0] = 0.0
        else:
            raise ValueError(f"Unknown orientation: {orientation}")

        norm = math.sqrt(sum(component * component for component in heading))
        if norm == 0.0 or math.isnan(norm):
            return math.inf, endpoint.position

        unit = tuple(component / norm for component in heading)
        delta = self._subtract(endpoint_target.position, endpoint.position)
        if orientation == "xz":
            multiplier = unit[0] * delta[0] + unit[2] * delta[2]
        else:
            multiplier = unit[1] * delta[1] + unit[2] * delta[2]

        extrapolated = self._add(endpoint.position, self._scale(unit, multiplier))
        dca = self._distance(extrapolated, endpoint_target.position)
        return dca, extrapolated

    def _single_endpoint_vertex(
        self,
        vertex_id: int,
        tracklet: Tracklet,
        endpoint: _FitEndpoint,
    ) -> Vertex:
        vertex = Vertex(vertex_id)
        vertex.add_tracklet(tracklet)
        vertex.extra_info.update(
            {
                "valid": True,
                "front_overlap_type": self.SINGLE_ENDPOINT,
                "back_overlap_type": self.SINGLE_ENDPOINT,
                "front_distance": 0.0,
                "back_distance": 0.0,
                "front_endpoints": [endpoint] if self._endpoint_matches_orientation(tracklet, endpoint, "xz") else [],
                "back_endpoints": [endpoint] if self._endpoint_matches_orientation(tracklet, endpoint, "yz") else [],
                "front_vertex_position": (
                    self._projected_vertex_position([endpoint], "xz")
                    if self._endpoint_matches_orientation(tracklet, endpoint, "xz")
                    else None
                ),
                "back_vertex_position": (
                    self._projected_vertex_position([endpoint], "yz")
                    if self._endpoint_matches_orientation(tracklet, endpoint, "yz")
                    else None
                ),
                "tracklets_time_ordered": True,
            }
        )
        return vertex

    def _invalid_vertex(self, vertex_id: int) -> Vertex:
        vertex = Vertex(vertex_id)
        vertex.extra_info["valid"] = False
        return vertex

    def _fit_endpoints(self, tracklet: Tracklet, orientation: str | None = None) -> list[_FitEndpoint]:
        endpoints: list[_FitEndpoint] = []
        for endpoint in tracklet.extra_info.get("fit_endpoints", []):
            fit_endpoint = _FitEndpoint(
                tracklet_id=tracklet.tracklet_id,
                endpoint_index=int(endpoint["endpoint_index"]),
                position=tuple(endpoint["position"]),
                linear_heading=tuple(endpoint["linear_heading"]),
            )
            if orientation is None:
                if (
                    self._endpoint_matches_orientation(tracklet, fit_endpoint, "xz")
                    or self._endpoint_matches_orientation(tracklet, fit_endpoint, "yz")
                ):
                    endpoints.append(fit_endpoint)
                continue

            if self._endpoint_matches_orientation(tracklet, fit_endpoint, orientation):
                endpoints.append(fit_endpoint)
        return endpoints

    def _set_overlap_endpoint_info(
        self,
        vertex: Vertex,
        vertex_side: str,
        overlap: dict[str, Any],
    ) -> None:
        if overlap["type"] == self.ORIENTATION_MISSING:
            return

        endpoints = [overlap["endpoint_1"], overlap["endpoint_2"]]
        endpoints = [endpoint for endpoint in endpoints if endpoint is not None]
        vertex.extra_info[f"{vertex_side}_endpoints"] = endpoints
        orientation = "xz" if vertex_side == "front" else "yz"
        vertex.extra_info[f"{vertex_side}_vertex_position"] = self._projected_vertex_position(endpoints, orientation)

    def _remove_vertex_endpoints(self, all_endpoints: set[_FitEndpoint], vertex: Vertex) -> None:
        for endpoint in vertex.extra_info.get("front_endpoints", []):
            all_endpoints.discard(endpoint)
        for endpoint in vertex.extra_info.get("back_endpoints", []):
            all_endpoints.discard(endpoint)

    def _is_valid(self, vertex: Vertex) -> bool:
        return bool(vertex.extra_info.get("valid", False))

    def _score(self, vertex: Vertex) -> float:
        score = 0.0
        if vertex.extra_info.get("front_overlap_type") != self.ORIENTATION_MISSING:
            score += float(vertex.extra_info.get("front_distance", 0.0))
        if vertex.extra_info.get("back_overlap_type") != self.ORIENTATION_MISSING:
            score += float(vertex.extra_info.get("back_distance", 0.0))
        return score

    def _tracklet_time(self, tracklet: Tracklet) -> float:
        raw_tracklet = tracklet.extra_info.get("raw_tracklet")
        if raw_tracklet is None:
            return math.inf
        return float(raw_tracklet.GetTime())

    def _projected_distance(self, endpoint_1: _FitEndpoint, endpoint_2: _FitEndpoint, orientation: str) -> float:
        if orientation == "xz":
            return math.sqrt(
                (endpoint_1.position[0] - endpoint_2.position[0]) ** 2
                + (endpoint_1.position[2] - endpoint_2.position[2]) ** 2
            )
        if orientation == "yz":
            return math.sqrt(
                (endpoint_1.position[1] - endpoint_2.position[1]) ** 2
                + (endpoint_1.position[2] - endpoint_2.position[2]) ** 2
            )
        raise ValueError(f"Unknown orientation: {orientation}")

    def _mean_position(self, endpoints: list[_FitEndpoint]) -> tuple[float, float, float] | None:
        if not endpoints:
            return None
        scale = 1.0 / len(endpoints)
        return (
            sum(endpoint.position[0] for endpoint in endpoints) * scale,
            sum(endpoint.position[1] for endpoint in endpoints) * scale,
            sum(endpoint.position[2] for endpoint in endpoints) * scale,
        )

    def _projected_vertex_position(
        self,
        endpoints: list[_FitEndpoint],
        orientation: str,
    ) -> tuple[float, float, float] | None:
        if not endpoints:
            return None
        scale = 1.0 / len(endpoints)
        mean_z = sum(endpoint.position[2] for endpoint in endpoints) * scale
        if orientation == "xz":
            mean_coord = sum(endpoint.position[0] for endpoint in endpoints) * scale
            return (mean_coord, 0.0, mean_z)
        if orientation == "yz":
            mean_coord = sum(endpoint.position[1] for endpoint in endpoints) * scale
            return (0.0, mean_coord, mean_z)
        raise ValueError(f"Unknown orientation: {orientation}")

    def _endpoint_matches_orientation(
        self,
        tracklet: Tracklet,
        endpoint: _FitEndpoint,
        orientation: str,
    ) -> bool:
        z_value = endpoint.position[2]
        coord_value = endpoint.position[0] if orientation == "xz" else endpoint.position[1]
        if not self._is_display_coordinate(z_value) or not self._is_display_coordinate(coord_value):
            return False

        z_bounds, coord_bounds = self._projected_hit_bounds(tracklet, orientation)
        if z_bounds is None or coord_bounds is None:
            return False

        z_low, z_high = z_bounds
        coord_low, coord_high = coord_bounds
        return (
            z_low - self.ENDPOINT_Z_MARGIN_MM <= z_value <= z_high + self.ENDPOINT_Z_MARGIN_MM
            and coord_low - self.ENDPOINT_COORD_MARGIN_MM <= coord_value <= coord_high + self.ENDPOINT_COORD_MARGIN_MM
        )

    def _projected_hit_bounds(
        self,
        tracklet: Tracklet,
        orientation: str,
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        expected_side = "front" if orientation == "xz" else "back"
        z_values = []
        coord_values = []
        for hit in tracklet.hits:
            if hit.detector_side != expected_side or hit.z is None:
                continue
            coord = hit.x if orientation == "xz" else hit.y
            if coord is None:
                continue
            z_values.append(float(hit.z))
            coord_values.append(float(coord))
        if not z_values or not coord_values:
            return None, None
        return (min(z_values), max(z_values)), (min(coord_values), max(coord_values))

    def _is_display_coordinate(self, value: float) -> bool:
        return math.isfinite(value) and abs(value) < 1.0e6

    def _distance(self, first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
        return math.sqrt(sum(component * component for component in self._subtract(first, second)))

    def _midpoint(self, first: tuple[float, float, float], second: tuple[float, float, float]) -> tuple[float, float, float]:
        return (
            0.5 * (first[0] + second[0]),
            0.5 * (first[1] + second[1]),
            0.5 * (first[2] + second[2]),
        )

    def _add(self, first: tuple[float, float, float], second: tuple[float, float, float]) -> tuple[float, float, float]:
        return (first[0] + second[0], first[1] + second[1], first[2] + second[2])

    def _subtract(self, first: tuple[float, float, float], second: tuple[float, float, float]) -> tuple[float, float, float]:
        return (first[0] - second[0], first[1] - second[1], first[2] - second[2])

    def _scale(self, vector: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
        return (vector[0] * scale, vector[1] * scale, vector[2] * scale)
