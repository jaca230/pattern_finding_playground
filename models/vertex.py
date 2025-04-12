from typing import Set, Optional
from models.tracklet import Tracklet

class Vertex:
    def __init__(self, vertex_id: int, tracklets: Optional[Set[Tracklet]] = None):
        self.tracklets: Set[Tracklet] = tracklets if tracklets is not None else set()
        self.vertex_id = vertex_id
        self.extra_info: dict = {}

    def get_tracklets(self) -> Set[Tracklet]:
        return self.tracklets

    def add_tracklet(self, tracklet: Tracklet) -> None:
        self.tracklets.add(tracklet)

    def __repr__(self) -> str:
        return f"Vertex(id={self.vertex_id}, num_tracklets={len(self.tracklets)})"

    def __eq__(self, other):
        if not isinstance(other, Vertex):
            return NotImplemented
        return self.tracklets == other.tracklets

    def __hash__(self):
        # Hash only on the tracklets, since that's how we're defining equality
        return hash(frozenset(self.tracklets))
