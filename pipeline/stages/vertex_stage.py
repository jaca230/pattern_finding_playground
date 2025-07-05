from pipeline.staging.stage import Stage
from algorithms.vertex.vertex_former import VertexFormer
from enums.default_stages import DefaultStages
from pipeline.staging.stage import monitor_stage_performance

class VertexStage(Stage):
    def __init__(self, vertex_former: VertexFormer):
        super().__init__(
            name="Form Vertices",
            stage_value=DefaultStages.VERTICES,
            prerequisites=[DefaultStages.TRACKLETS]
        )
        self.vertex_former = vertex_former

    def build_handler(self):
        @monitor_stage_performance(stage_name=self.name)
        def handler(storage):
            # Direct reference - no copying
            tracklets = storage["tracklets"]
            
            # Call the algorithm
            vertices, info = self.vertex_former.form_vertices(tracklets, storage=storage)

            # Direct assignment - vertices are likely already optimized collections
            storage["vertices"] = vertices
            
            # Use setdefault to avoid double lookup
            storage.setdefault("extra_info", {})["vertex_algorithm_info"] = info
        return handler