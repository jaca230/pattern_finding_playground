from pipeline.staging.stage import Stage
from algorithms.tracklet.tracklet_former import TrackletFormer
from enums.default_stages import DefaultStages
from pipeline.staging.stage import monitor_stage_performance

class TrackletStage(Stage):
    def __init__(self, tracklet_former: TrackletFormer):
        super().__init__(
            name="Form Tracklets",
            stage_value=DefaultStages.TRACKLETS,
            prerequisites=[DefaultStages.INPUT]
        )
        self.tracklet_former = tracklet_former

    def build_handler(self):
        @monitor_stage_performance(stage_name=self.name)
        def handler(storage):
            # Cache frequently accessed items
            tree = storage["tree"]
            geo = storage["geo"]
            entry = storage["entry_index"]
            
            # Call the algorithm
            tracklets, info = self.tracklet_former.form_tracklets(tree, geo, entry, storage=storage)

            # Optimize set operations
            if "tracklets" not in storage:
                storage["tracklets"] = set(tracklets)  # Create with initial data
            else:
                storage["tracklets"].update(tracklets)  # Batch update

            # Use setdefault to avoid double lookup
            storage.setdefault("extra_info", {})["tracklet_algorithm_info"] = info
        return handler