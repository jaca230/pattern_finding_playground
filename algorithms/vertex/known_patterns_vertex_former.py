from typing import Set
from models.tracklet import Tracklet
from models.vertex import Vertex
from algorithms.vertex.vertex_former import VertexFormer  # adjust import path as needed

class KnownPatternsVertexFormer(VertexFormer):
    def form_vertices(self, tracklets: Set[Tracklet]) -> tuple[Set[Vertex], dict]:
        # Initialize an empty set to hold the vertices
        vertices = set()

        # Create a separate vertex for each tracklet
        for i, tracklet in enumerate(tracklets):
            vertex = Vertex(vertex_id=i)
            vertex.add_tracklet(tracklet)
            vertices.add(vertex)

        result_info = {}

        return vertices, result_info
