from enum import Enum
from typing import Set, List
from models.pattern import Pattern
from models.vertex import Vertex
from algorithms.vertex.vertex_former import VertexFormer
from algorithms.pattern.pattern_former import PatternFormer
from algorithms.tracklet.tracklet_former import TrackletFormer

from enum import Enum

class Stage(Enum):
    INIT = 0
    TRACKLETS_FORMED = 1
    VERTICES_FORMED = 2
    PATTERNS_FORMED = 3

    # Less than operator
    def __lt__(self, other):
        return self.value < other.value

    # Less than or equal operator
    def __le__(self, other):
        return self.value <= other.value

    # Greater than operator
    def __gt__(self, other):
        return self.value > other.value

    # Greater than or equal operator
    def __ge__(self, other):
        return self.value >= other.value

    # Equal operator
    def __eq__(self, other):
        return self.value == other.value


class EventPatterns:
    def __init__(self, event_id: int, tracklet_former: TrackletFormer, vertex_former: VertexFormer, pattern_former: PatternFormer):
        self.event_id = event_id
        self.tracklet_former = tracklet_former
        self.vertex_former = vertex_former
        self.pattern_former = pattern_former
        self.patterns: Set[Pattern] = set()
        self.extra_info: dict = {"stage": Stage.INIT}

    def form_tracklets(self, file, entry_index: int) -> None:
        if self.extra_info["stage"] < Stage.INIT:
            raise RuntimeError("Cannot form tracklets in the current stage.")
        
        tracklets, algorithm_info = self.tracklet_former.form_tracklets(file, entry_index)
        vertex = Vertex(0, set(tracklets))
        self.patterns = {Pattern(0, {vertex})}
        self.extra_info["tracklet_algorithm_info"] = algorithm_info
        self.extra_info["stage"] = Stage.TRACKLETS_FORMED

    def form_vertices(self) -> None:
        if self.extra_info["stage"] < Stage.TRACKLETS_FORMED:
            raise RuntimeError("Cannot form vertices unless tracklets are formed.")
        
        all_tracklets = {t for p in self.patterns for v in p.get_vertices() for t in v.get_tracklets()}
        vertices, algorithm_info = self.vertex_former.form_vertices(all_tracklets)
        self.patterns = {Pattern(0, vertices)}
        self.extra_info["vertex_algorithm_info"] = algorithm_info
        self.extra_info["stage"] = Stage.VERTICES_FORMED

    def form_patterns(self) -> None:
        if self.extra_info["stage"] < Stage.VERTICES_FORMED:
            raise RuntimeError("Cannot form patterns unless vertices are formed.")
        
        all_vertices = {v for p in self.patterns for v in p.get_vertices()}
        self.patterns, algorithm_info = self.pattern_former.form_patterns(all_vertices)
        self.extra_info["pattern_algorithm_info"] = algorithm_info
        self.extra_info["stage"] = Stage.PATTERNS_FORMED

    def form_all(self, file, entry_index: int) -> None:
        """Calls all three formers in sequence."""
        self.form_tracklets(file, entry_index)
        self.form_vertices()
        self.form_patterns()

    def get_patterns(self) -> Set[Pattern]:
        return self.patterns

    def add_pattern(self, pattern: Pattern) -> None:
        self.patterns.add(pattern)

    def __repr__(self) -> str:
        return f"EventPatterns(num_patterns={len(self.patterns)}, stage={self.extra_info['stage']})"
