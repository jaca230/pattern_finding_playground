import math
from typing import Any, Tuple, Optional
from models.vertex import Vertex
from models.tracklet import Tracklet
from algorithms.vertex.vertex_types.scoring.vertex_scorer import VertexScorer


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

        seed_endpoints = [pt for pt in seed.get_endpoints() if pt is not None]
        if not seed_endpoints:
            raise ValueError("Seed tracklet is missing both endpoints.")

        closest_tracklet = None
        min_distance = float("inf")

        for other in candidates:
            if other is seed:
                continue

            other_endpoints = [pt for pt in other.get_endpoints() if pt is not None]
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
