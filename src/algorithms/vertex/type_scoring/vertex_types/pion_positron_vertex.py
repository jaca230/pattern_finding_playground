from algorithms.vertex.type_scoring.vertex_scorers.distance_scorer import DistanceScorer
from algorithms.vertex.type_scoring.vertex_types.vertex_type import VertexType


class PionPositronVertex(VertexType):
    def __init__(self):
        super().__init__(
            id="pi+_to_e+",
            input_particles={211},
            output_particles={-11},
            scorer=DistanceScorer(),
        )
