from typing import Optional, Set, Any, Tuple
from models.tracklet import Tracklet
from models.vertex import Vertex
from abc import ABC, abstractmethod

class VertexFormer(ABC):
    @abstractmethod
    def form_vertices(self, tracklets: Set[Tracklet], storage: Optional[Any] = None) -> Tuple[Set[Vertex], dict]:
        pass
