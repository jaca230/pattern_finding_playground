from abc import ABC, abstractmethod
from typing import Set
from models.tracklet import Tracklet
from models.vertex import Vertex

class VertexFormer(ABC):
    @abstractmethod
    def form_vertices(self, tracklets: Set[Tracklet]) -> Set[Vertex]:
        pass

