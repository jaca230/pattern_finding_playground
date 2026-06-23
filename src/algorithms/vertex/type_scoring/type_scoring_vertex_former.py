from typing import Optional, Set, Any, Tuple, List
from algorithms.registry import register_algorithm
from algorithms.vertex.type_scoring.vertex_types import (
    MuonPositronVertex,
    PionMuonVertex,
    PionPositronVertex,
    VertexType,
)
from algorithms.vertex.vertex_former import VertexFormer
from models.tracklet import Tracklet
from models.vertex import Vertex


@register_algorithm(
    "vertex",
    name="type_scoring",
    description="Score allowed vertex particle hypotheses and keep the best-scoring vertex for each seed tracklet.",
)
class TypeScoringVertexFormer(VertexFormer):
    def __init__(self, vertex_types: Optional[List[VertexType]] = None):
        self.vertex_types = vertex_types or self._default_vertex_types()
        self.vertex_id_counter = 0

    def _default_vertex_types(self) -> list[VertexType]:
        return [PionMuonVertex(), PionPositronVertex(), MuonPositronVertex()]

    def form_vertices(self, tracklets: Set[Tracklet], storage: Optional[Any] = None) -> Tuple[Set[Vertex], dict]:
        formed_vertices: Set[Vertex] = set()

        for seed_tracklet in tracklets:
            best_vertex: Optional[Vertex] = None
            best_score: float = float("-inf")

            for vertex_type in self.vertex_types:
                if seed_tracklet.particle_id not in vertex_type.input_particles:
                    continue

                context = {
                    "seed_tracklet": seed_tracklet,
                    "tracklets": tracklets,
                    "vertices": formed_vertices,
                    "vertex_id": self.vertex_id_counter,
                }

                score = vertex_type.score(context)

                if vertex_type.vertex is not None and score > best_score:
                    best_score = score
                    best_vertex = vertex_type.vertex

            if best_vertex is not None:
                self.vertex_id_counter += 1
                best_vertex.vertex_id = self.vertex_id_counter
                formed_vertices.add(best_vertex)

        result_info = {}

        return formed_vertices, result_info
