from algorithms.vertex.type_scoring.vertex_scorers.distance_scorer import DistanceScorer
from algorithms.vertex.type_scoring.vertex_types.vertex_type import VertexType


class PionMuonVertex(VertexType):
    def __init__(self):
        super().__init__(
            id="pi+_to_mu+",
            input_particles={211},
            output_particles={-13},
            scorer=DistanceScorer(),
        )
