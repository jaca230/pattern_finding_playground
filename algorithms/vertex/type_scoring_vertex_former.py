from typing import Optional, Set, Any, Tuple, List
from models.tracklet import Tracklet
from models.vertex import Vertex
from algorithms.vertex.vertex_types.vertex_type import VertexType
from algorithms.vertex.vertex_types.vertex_types import PionMuonVertex, PionPositronVertex, MuonPositronVertex
from algorithms.vertex.vertex_former import VertexFormer


class TypeScoringVertexFormer(VertexFormer):
    def __init__(self, vertex_types: Optional[List[VertexType]] = None):
        # If none provided, load a default set
        self.vertex_types = vertex_types or self._default_vertex_types()
        self.vertex_id_counter = 0  # Initialize a counter for vertex IDs

    def _default_vertex_types(self) -> tuple[Set[Vertex], dict]:
        # Use the predefined vertex types as the default
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
                # Increment the vertex_id counter for the next vertex
                self.vertex_id_counter += 1
                best_vertex.vertex_id = self.vertex_id_counter  # Update the vertex ID
                formed_vertices.add(best_vertex)

        result_info = {}

        return formed_vertices, result_info
