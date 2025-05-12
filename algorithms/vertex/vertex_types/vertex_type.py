from typing import Set, Any
from models.vertex import Vertex
from algorithms.vertex.vertex_types.scoring.vertex_scorer import VertexScorer


class VertexType:
    def __init__(
        self,
        id: str,
        input_particles: Set[int],
        output_particles: Set[int],
        scorer: VertexScorer,
    ):
        self._id = id
        self._input_particles = input_particles
        self._output_particles = output_particles
        self._scorer = scorer
        self._vertex: Vertex | None = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def input_particles(self) -> Set[int]:
        return self._input_particles

    @property
    def output_particles(self) -> Set[int]:
        return self._output_particles

    @property
    def scorer(self) -> VertexScorer:
        return self._scorer

    @property
    def vertex(self) -> Vertex:
        return self._vertex

    def _cleanse_context(self, context: dict) -> dict:
        if "tracklets" not in context:
            raise ValueError("Context must contain 'tracklets'.")

        filtered = [
            t for t in context["tracklets"]
            if t.particle_id in self._output_particles
        ]

        new_context = context.copy()
        new_context["tracklets"] = filtered
        return new_context

    def score(self, context: dict) -> float:
        cleansed_context = self._cleanse_context(context)
        score, vertex = self._scorer.score(cleansed_context)
        self._vertex = vertex
        return score
