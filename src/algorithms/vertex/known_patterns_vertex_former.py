from typing import Set, Any, Optional, Tuple
import numpy as np

from algorithms.registry import register_algorithm
from models.tracklet import Tracklet
from models.vertex import Vertex
from algorithms.vertex.vertex_former import VertexFormer  # adjust import path as needed

@register_algorithm(
    "vertex",
    name="known_patterns",
    description="Create one display vertex per tracklet using the tracklet's hit positions.",
)
class KnownPatternsVertexFormer(VertexFormer):
    def form_vertices(self, tracklets: Set[Tracklet], storage: Optional[Any] = None) -> Tuple[Set[Vertex], dict]:
        # Initialize an empty set to hold the vertices
        vertices = set()

        # Create a separate vertex for each tracklet
        for i, tracklet in enumerate(tracklets):
            vertex = Vertex(vertex_id=i)
            vertex.add_tracklet(tracklet)
            self._attach_display_positions(vertex, tracklet)
            vertices.add(vertex)

        result_info = {}

        return vertices, result_info

    def _attach_display_positions(self, vertex: Vertex, tracklet: Tracklet) -> None:
        front_points = [(hit.x, hit.z) for hit in tracklet.get_front_hits() if hit.x is not None and hit.z is not None]
        back_points = [(hit.y, hit.z) for hit in tracklet.get_back_hits() if hit.y is not None and hit.z is not None]

        if front_points:
            arr = np.asarray(front_points, dtype=float)
            vertex.extra_info["front_vertex_position"] = (float(np.mean(arr[:, 0])), 0.0, float(np.mean(arr[:, 1])))
        if back_points:
            arr = np.asarray(back_points, dtype=float)
            vertex.extra_info["back_vertex_position"] = (0.0, float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))
        if front_points or back_points:
            vertex.extra_info["valid"] = True
