"""Default stages for backward compatibility - users can define their own."""
from enum import Enum, auto

class DefaultStages(Enum):
    INPUT = auto()
    EVENT_INIT = auto()
    TRACKLETS = auto()
    VERTICES = auto()
    PATTERNS = auto()
    VALIDATION = auto()