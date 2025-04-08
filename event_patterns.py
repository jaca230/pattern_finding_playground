# event_patterns.py

from typing import Set
from pattern import Pattern


class EventPatterns:
    def __init__(self, patterns: Set[Pattern]):
        """
        Represents an event-level collection of patterns.

        Args:
            patterns: A set of Pattern objects that make up the event.
        """
        self.patterns = patterns

    def get_patterns(self) -> Set[Pattern]:
        """Returns the set of patterns in the event."""
        return self.patterns

    def add_pattern(self, pattern: Pattern) -> None:
        """Adds a pattern to the event."""
        self.patterns.add(pattern)

    def __repr__(self) -> str:
        return f"EventPatterns(num_patterns={len(self.patterns)})"
