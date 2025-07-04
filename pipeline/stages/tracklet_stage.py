from pipeline.staging.stage import Stage
from algorithms.tracklet.tracklet_former import TrackletFormer
from enums.default_stages import DefaultStages

class TrackletStage(Stage):
    def __init__(self, tracklet_former: TrackletFormer):
        super().__init__(
            name="Form Tracklets",
            stage_value=DefaultStages.TRACKLETS,
            prerequisites=[DefaultStages.INPUT]
        )
        self.tracklet_former = tracklet_former

    def build_handler(self):
        def handler(storage):
            tree = storage["tree"]
            geo = storage["geo"]
            entry = storage["entry_index"]
            tracklets, info = self.tracklet_former.form_tracklets(tree, geo, entry, storage=storage)

            # Initialize the set if missing
            if "tracklets" not in storage:
                storage["tracklets"] = set()
            # Clear and update in place so references remain valid
            storage["tracklets"].update(tracklets)

            storage.setdefault("extra_info", {})["tracklet_algorithm_info"] = info
        return handler
