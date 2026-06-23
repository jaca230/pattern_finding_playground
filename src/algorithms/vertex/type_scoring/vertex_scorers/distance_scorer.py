import math
from typing import Any, Tuple, Optional
from models.vertex import Vertex
from models.tracklet import Tracklet
from models.point_3d import Point3D
from algorithms.vertex.type_scoring.vertex_scorers.vertex_scorer import VertexScorer


class DistanceScorer(VertexScorer):
    """
    Scores a vertex by the proximity of the closest pair of endpoints between the seed_tracklet
    and any other tracklet. The vertex consists of the seed and the closest matching tracklet.
    Score is calculated as exp(-distance), so it lies in (0, 1].
    If no closest tracklet is found, a vertex with just the seed tracklet is returned, with score 0.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def score(self, context: Any) -> Tuple[float, Optional[Vertex]]:
        # Extract the vertex_id from the context
        vertex_id: int = context.get("vertex_id", -1)  # Default to -1 if no vertex_id is provided
        
        seed: Tracklet = context["seed_tracklet"]
        candidates: list[Tracklet] = context["tracklets"]

        seed_endpoints = self._endpoints_for_tracklet(seed)
        if not seed_endpoints:
            raise ValueError("Seed tracklet is missing both endpoints.")

        closest_tracklet = None
        min_distance = float("inf")

        for other in candidates:
            if other is seed:
                continue

            other_endpoints = self._endpoints_for_tracklet(other)
            if not other_endpoints:
                continue

            for pt_seed in seed_endpoints:
                for pt_other in other_endpoints:
                    dist = pt_seed.distance_to(pt_other)
                    if dist < min_distance:
                        min_distance = dist
                        closest_tracklet = other

        if closest_tracklet is None:
            # Return a vertex with just the seed and score 0
            return 0.0, Vertex(vertex_id=vertex_id, tracklets={seed})

        score = math.exp(-self.alpha * min_distance)
        vertex = Vertex(vertex_id=vertex_id, tracklets={seed, closest_tracklet})
        return score, vertex

    def _endpoints_for_tracklet(self, tracklet: Tracklet) -> list[Point3D]:
        endpoints = [point for point in tracklet.get_endpoints() if point is not None]
        if endpoints:
            return endpoints

        inferred = self._infer_endpoints_from_hits(tracklet)
        if len(inferred) == 2:
            tracklet.set_endpoints(inferred[0], inferred[1])
        return inferred

    def _infer_endpoints_from_hits(self, tracklet: Tracklet) -> list[Point3D]:
        valid_hits = [hit for hit in tracklet.hits if hit.z is not None]
        if not valid_hits:
            return []

        ordered_hits = sorted(valid_hits, key=lambda hit: float(hit.z))
        first_hit = ordered_hits[0]
        last_hit = ordered_hits[-1]

        start_point = self._point_from_hits_near_z(ordered_hits, float(first_hit.z))
        stop_point = self._point_from_hits_near_z(ordered_hits, float(last_hit.z))

        if start_point is None and stop_point is None:
            return []
        if start_point is None:
            return [stop_point]
        if stop_point is None:
            return [start_point]
        return [start_point, stop_point]

    def _point_from_hits_near_z(self, hits, target_z: float, tolerance_mm: float = 1.0e-6) -> Optional[Point3D]:
        nearby_hits = [hit for hit in hits if abs(float(hit.z) - target_z) <= tolerance_mm]
        if not nearby_hits:
            nearby_hits = [min(hits, key=lambda hit: abs(float(hit.z) - target_z))]

        x_values = [float(hit.x) for hit in nearby_hits if hit.x is not None]
        y_values = [float(hit.y) for hit in nearby_hits if hit.y is not None]

        x = sum(x_values) / len(x_values) if x_values else 0.0
        y = sum(y_values) / len(y_values) if y_values else 0.0
        return Point3D(x=x, y=y, z=target_z)
