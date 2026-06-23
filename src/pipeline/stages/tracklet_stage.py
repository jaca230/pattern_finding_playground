from pipeline.stages.stage import Stage
from algorithms.tracklet.tracklet_former import TrackletFormer

class TrackletStage(Stage):
    def __init__(self, tracklet_former: TrackletFormer):
        super().__init__(
            name="Form Tracklets",
            stage_key="tracklets",
            prerequisites=["input"]
        )
        self.tracklet_former = tracklet_former

    def build_handler(self):
        def handler(storage):
            event_entry = storage["event_entry"]
            reference_truth_entry = storage.get("reference_truth_entry")
            geo = storage["geo"]

            tracklets, info = self.tracklet_former.form_tracklets(
                event_entry,
                geo,
                storage=storage,
                reference_truth_entry=reference_truth_entry,
            )

            # Optimize set operations
            if "tracklets" not in storage:
                storage["tracklets"] = set(tracklets)  # Create with initial data
            else:
                storage["tracklets"].update(tracklets)  # Batch update

            # Use setdefault to avoid double lookup
            storage.setdefault("extra_info", {})["tracklet_algorithm_info"] = info
        return handler
