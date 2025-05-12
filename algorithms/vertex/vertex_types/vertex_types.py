# vertex_types.py

from algorithms.vertex.vertex_types.vertex_type import VertexType
from algorithms.vertex.vertex_types.scoring.distance_scorer import DistanceScorer


class PionMuonVertex(VertexType):
    def __init__(self):
        super().__init__(
            id="pi+_to_mu+",
            input_particles={211},  # pi+
            output_particles={-13},  # mu+
            scorer=DistanceScorer()
        )


class PionPositronVertex(VertexType):
    def __init__(self):
        super().__init__(
            id="pi+_to_e+",
            input_particles={211},  # pi+
            output_particles={-11},  # e+
            scorer=DistanceScorer()
        )


class MuonPositronVertex(VertexType):
    def __init__(self):
        super().__init__(
            id="mu+_to_e+",
            input_particles={-13},  # mu+
            output_particles={-11},  # e+
            scorer=DistanceScorer()
        )
