"""Default stages users can insert/define their own stages in between"""
from enum import Enum

class DefaultStages(Enum):
    INPUT = 100
    EVENT_INIT = 200
    TRACKLETS = 300
    VERTICES = 400
    PATTERNS = 500
    VALIDATION = 600