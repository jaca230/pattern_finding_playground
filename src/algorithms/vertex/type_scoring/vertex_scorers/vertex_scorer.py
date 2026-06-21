from abc import ABC, abstractmethod
from typing import Any, Tuple
from models.vertex import Vertex  # or relative path if needed

class VertexScorer(ABC):
    @abstractmethod
    def score(self, context: Any) -> Tuple[float, Vertex]:
        """
        Computes a score and constructs a vertex given the context.
        Returns:
            (score, vertex) where:
                score (float): likelihood or quality of this vertex hypothesis.
                vertex (Vertex): constructed vertex object.
        """
        pass
