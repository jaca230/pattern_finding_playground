from pipeline.staging.stage import Stage
from algorithms.vertex.vertex_former import VertexFormer
from enums.default_stages import DefaultStages

class VertexStage(Stage):
    def __init__(self, vertex_former: VertexFormer):
        super().__init__(
            name="Form Vertices",
            stage_value=DefaultStages.VERTICES,
            prerequisites=[DefaultStages.TRACKLETS]
        )
        self.vertex_former = vertex_former

    def build_handler(self):
        def handler(storage):
            tracklets = storage["tracklets"]
            vertices, info = self.vertex_former.form_vertices(tracklets, storage=storage)

            storage["vertices"] = vertices
            storage.setdefault("extra_info", {})["vertex_algorithm_info"] = info
        return handler

