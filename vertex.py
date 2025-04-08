from typing import Set
from tracklet import Tracklet


class Vertex:
    def __init__(self, seed_tracklet: Tracklet, vertex_id: int):
        """
        Represents a vertex formed by a group of tracklets.

        Args:
            seed_tracklet: A Tracklet object to serve as the starting point for the vertex.
            vertex_id: Unique identifier for the vertex.
        """
        # Initialize the vertex with a set containing the seed tracklet
        self.tracklets = {seed_tracklet}
        self.vertex_id = vertex_id  # Store the unique ID

    def get_tracklets(self) -> Set[Tracklet]:
        """Returns the set of tracklets associated with this vertex."""
        return self.tracklets

    def add_tracklet(self, tracklet: Tracklet) -> None:
        """Adds a tracklet to the vertex."""
        self.tracklets.add(tracklet)

    def __repr__(self) -> str:
        return f"Vertex(id={self.vertex_id}, num_tracklets={len(self.tracklets)})"
