from algorithms.vertex.type_scoring.vertex_scorers.distance_scorer import DistanceScorer
from algorithms.vertex.type_scoring.vertex_types.vertex_type import VertexType


class MuonPositronVertex(VertexType):
    def __init__(self):
        super().__init__(
            id="mu+_to_e+",
            input_particles={-13},
            output_particles={-11},
            scorer=DistanceScorer(),
        )
