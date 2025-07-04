from typing import Set, Optional, Any
from models.pattern import Pattern
from models.vertex import Vertex
from algorithms.pattern.pattern_former import PatternFormer


class KnownPatternsPatternFormer(PatternFormer):
    def __init__(self):
        """KnownPatternsPatternFormer forms patterns based on the 'pattern_reco' information stored in the tracklet's extra_info."""
        pass

    def form_patterns(self, vertices: Set[Vertex], storage: Optional[Any] = None) -> tuple[Set[Pattern], dict]:
        """
        Forms patterns based on the 'pattern_reco' information in tracklet's extra_info.
        Each tracklet’s 'pattern_reco' extra_info indicates which pattern it belongs to.
        """
        patterns = set()
        pattern_groups = {}

        # Group tracklets by their 'pattern_reco' extra_info
        for vertex in vertices:
            tracklets = vertex.get_tracklets()

            if len(tracklets) != 1:
                raise ValueError(f"Vertex {vertex} contains {len(tracklets)} tracklets. Expected exactly one.")

            # Get the only tracklet from this vertex
            tracklet = next(iter(tracklets))
            pattern_id = tracklet.extra_info.get("pattern_reco", None)

            if pattern_id is not None:
                if pattern_id not in pattern_groups:
                    pattern_groups[pattern_id] = set()
                pattern_groups[pattern_id].add(vertex)

        # Create patterns from the grouped vertices
        for pattern_id, grouped_vertices in pattern_groups.items():
            patterns.add(Pattern(pattern_id, grouped_vertices))

        result_info = {"num_patterns": len(patterns)}
        return patterns, result_info
