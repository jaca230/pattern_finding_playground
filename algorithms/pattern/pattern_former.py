from abc import ABC, abstractmethod
from typing import Set
from models.pattern import Pattern
from models.vertex import Vertex


class PatternFormer(ABC):
    @abstractmethod
    def form_patterns(self, vertices: Set[Vertex]) -> tuple[Set[Pattern], dict]:
        pass
