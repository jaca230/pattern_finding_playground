from .vertex_scorer import VertexScorer
from models.vertex import Vertex

class DummyScorer(VertexScorer):
    def score(self, context) -> tuple[float, Vertex]:
        vertex = Vertex(vertex_id=0, tracklets=set())  # Dummy for testing
        return 1.0, vertex
