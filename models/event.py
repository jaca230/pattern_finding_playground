from typing import Set, Dict, Any
from models.pattern import Pattern
from models.tracklet import Tracklet  # Assuming you have this class

class Event:
    def __init__(self, event_id: int):
        self.event_id: int = event_id
        self.patterns: Set[Pattern] = set()
        self.all_tracklets: Set[Tracklet] = set()  # <-- New attribute for all tracklets
        self.extra_info: Dict[str, Any] = {}
        self.is_valid: bool = False  # Indicates event validity

    def get_patterns(self) -> Set[Pattern]:
        return self.patterns

    def add_pattern(self, pattern: Pattern) -> None:
        self.patterns.add(pattern)

    def set_patterns(self, patterns: Set[Pattern]) -> None:
        self.patterns = patterns

    def __repr__(self) -> str:
        pattern_ids = [getattr(p, 'id', repr(p)) for p in list(self.patterns)[:3]]
        tracklet_ids = [getattr(t, 'tracklet_id', repr(t)) for t in list(self.all_tracklets)[:3]]

        patterns_summary = (f"{len(self.patterns)} patterns (sample IDs: {pattern_ids})"
                            if self.patterns else "no patterns")
        tracklets_summary = (f"{len(self.all_tracklets)} tracklets (sample IDs: {tracklet_ids})"
                            if self.all_tracklets else "no tracklets")

        extra_info_keys = list(self.extra_info.keys())
        if len(extra_info_keys) > 5:
            extra_info_summary = f"{len(extra_info_keys)} keys (sample: {extra_info_keys[:5]})"
        else:
            extra_info_summary = f"{len(extra_info_keys)} keys: {extra_info_keys}"

        return (f"Event(id={self.event_id}, {patterns_summary}, "
                f"{tracklets_summary}, extra_info: {extra_info_summary}, "
                f"is_valid={self.is_valid})")

