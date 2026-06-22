from pipeline.stages.stage import Stage
from algorithms.vertex.vertex_former import VertexFormer

class VertexStage(Stage):
    def __init__(self, vertex_former: VertexFormer):
        super().__init__(
            name="Form Vertices",
            stage_key="vertices",
            prerequisites=["tracklets"]
        )
        self.vertex_former = vertex_former

    def build_handler(self):
        def handler(storage):
            # Direct reference - no copying
            tracklets = storage["tracklets"]
            
            # Call the algorithm
            vertices, info = self.vertex_former.form_vertices(tracklets, storage=storage)

            # Keep the storage set stable so Event.all_vertices remains live
            # during partial-stage notebook displays.
            storage_vertices = storage.setdefault("vertices", set())
            storage_vertices.clear()
            storage_vertices.update(vertices)
            
            # Use setdefault to avoid double lookup
            storage.setdefault("extra_info", {})["vertex_algorithm_info"] = info
        return handler
