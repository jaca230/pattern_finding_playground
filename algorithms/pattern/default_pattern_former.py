from typing import Set
from models.pattern import Pattern
from models.vertex import Vertex
from algorithms.pattern.pattern_former import PatternFormer


class DefaultPatternFormer(PatternFormer):
    def __init__(self):
        """DefaultPatternFormer uses connected component analysis based on shared tracklets."""
        pass

    def form_patterns(self, vertices: Set[Vertex]) -> tuple[Set[Pattern], dict]:
        """
        Forms patterns by identifying connected components in the vertex graph.
        Two vertices are connected if they share at least one tracklet.
        """
        patterns = set()
        unvisited = set(vertices)
        pattern_id = 0  # Start pattern ID counter

        while unvisited:  # Visit each vertex in the set exactly once
            start = unvisited.pop()  # start at some vertex in the set
            component = self._dfs(start, unvisited)  # find all connected vertices
            patterns.add(Pattern(pattern_id, component))  # add the connected component as a pattern
            pattern_id += 1

        result_info = {}

        return patterns, result_info

    def _dfs(self, start: Vertex, unvisited: Set[Vertex]) -> Set[Vertex]:
        """Performs depth-first search to find all vertices connected to 'start'."""
        stack = [start]
        connected = set()
        connected.add(start) # add the starting vertex to the connected set

        while stack: # This while loop avoids recursion depth issues
            current = stack.pop() # removes the vertex we're currently visitin from the "to visit" stack
            to_link = {v for v in unvisited if not current.tracklets.isdisjoint(v.tracklets)} #find links to this vertex
            stack.extend(to_link) # adds new vertices to visit
            connected.update(to_link) # add new vertices to connected set
            unvisited.difference_update(to_link) # removes vertices we're going to visit from the unvisted set

        return connected
