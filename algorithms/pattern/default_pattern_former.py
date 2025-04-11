from typing import Set
from models.pattern import Pattern
from models.vertex import Vertex
from algorithms.pattern.pattern_former import PatternFormer

class DefaultPatternFormer(PatternFormer):
    def __init__(self):
        """Initialize the DefaultPatternFormer (no specific initialization required yet)."""
        pass

    def form_patterns(self, vertices: Set[Vertex]) -> Set[Pattern]:
        patterns = set()
        visited = set()

        # Loop over all vertices and create patterns for each link group
        for vertex in vertices:
            if vertex not in visited:
                visited.add(vertex)
                linked_vertices = self._find_linked_vertices(vertex, visited, vertices)
                pattern = Pattern(linked_vertices)
                patterns.add(pattern)

        return patterns

    def _find_linked_vertices(self, start_vertex: Vertex, visited: Set[Vertex], vertices: Set[Vertex]) -> Set[Vertex]:
        """
        Private helper method to find all linked vertices for a given vertex.

        Args:
            start_vertex: The vertex from which to start the search.
            visited: A set of already visited vertices to avoid cycles.
            vertices: A set of all vertices available to be linked.

        Returns:
            A set of linked vertices.
        """
        linked_vertices = set([start_vertex])
        to_visit = [start_vertex]

        while to_visit:
            vertex = to_visit.pop()

            # Loop through all other vertices and check if they're linked through shared tracklets
            for other_vertex in vertices:
                if other_vertex != vertex and other_vertex not in visited:
                    # Check for shared tracklets between vertex and other_vertex
                    if self._are_linked(vertex, other_vertex):
                        visited.add(other_vertex)
                        linked_vertices.add(other_vertex)
                        to_visit.append(other_vertex)

        return linked_vertices

    def _are_linked(self, vertex_1: Vertex, vertex_2: Vertex) -> bool:
        """
        Private helper method to check if two vertices are linked by sharing tracklets.

        Args:
            vertex_1: The first vertex to check.
            vertex_2: The second vertex to check.

        Returns:
            True if the vertices share at least one tracklet, False otherwise.
        """
        # Assuming that vertices have a `tracklets` attribute that stores all tracklets they are part of
        return not vertex_1.tracklets.isdisjoint(vertex_2.tracklets)
