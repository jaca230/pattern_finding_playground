from typing import Set
from models.vertex import Vertex
from models.tracklet import Tracklet


class Pattern:
    def __init__(self, vertices: Set[Vertex]):
        """
        Represents a full event-level pattern composed of vertices.

        Args:
            vertices: A set of Vertex objects that make up the pattern.
        """
        self.vertices = vertices

    def get_vertices(self) -> Set[Vertex]:
        """Returns the set of vertices in the pattern."""
        return self.vertices

    def add_vertex(self, vertex: Vertex) -> None:
        """Adds a vertex to the pattern."""
        self.vertices.add(vertex)

    def get_unique_tracklets(self) -> Set[Tracklet]:
        """Returns a set of unique tracklets from all vertices in the pattern."""
        unique_tracklets = set()
        for vertex in self.vertices:
            unique_tracklets.update(vertex.get_tracklets())
        return unique_tracklets

    def __repr__(self) -> str:
        return f"Pattern(num_vertices={len(self.vertices)})"
