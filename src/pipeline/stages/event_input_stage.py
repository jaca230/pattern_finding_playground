from pipeline.stages.stage import Stage
from models.hit import Hit

class EventInputStage(Stage):
    def __init__(self):
        super().__init__(
            name="Load Event Input Data",
            stage_key="input",
            prerequisites=[],
            is_input_stage=True
        )

    def build_handler(self):
        def handler(storage, input_context):
            storage["event_data"] = input_context.event_data
            storage["truth_data"] = input_context.truth_data
            storage["geo"] = input_context.event_data.geo
            storage.setdefault("extra_info", {})["geo"] = input_context.event_data.geo
            storage["entry_index"] = input_context.entry_index
            storage["event_entry"] = input_context.event_data.load_entry(input_context.entry_index)
            storage["raw_hits"] = self._build_raw_hits(storage["event_entry"], storage["geo"])
            if not input_context.use_truth:
                storage["reference_truth_entry"] = None
            elif input_context.truth_data is not None:
                storage["reference_truth_entry"] = input_context.truth_data.load_entry(input_context.entry_index)
            else:
                storage["reference_truth_entry"] = storage["event_entry"].get("truth_entry")
        return handler

    def _build_raw_hits(self, event_entry, geo):
        hits = []
        for hit in event_entry["hits"]:
            vid = hit.GetVID()
            volume_name = geo.GetVolumeName(vid).Data()
            if "atar" not in volume_name:
                continue

            detector_side = None
            if len(volume_name) > 11:
                if volume_name[11] == "f":
                    detector_side = "front"
                elif volume_name[11] == "b":
                    detector_side = "back"

            hits.append(
                Hit(
                    z=geo.GetZ(vid) + 0.07,
                    x=geo.GetX(vid),
                    y=geo.GetY(vid),
                    time=hit.GetObservedTime(),
                    energy=hit.GetObservedEdep(),
                    particle_id=hit.GetPID(),
                    detector_side=detector_side,
                    volume_id=int(vid),
                    volume_name=volume_name,
                )
            )
        return hits

# Simple class to hold inputs for EventInputStage runs
class InputContext:
    __slots__ = ("event_data", "entry_index", "truth_data", "use_truth")

    def __init__(self, event_data, entry_index, truth_data=None, use_truth=True):
        self.event_data = event_data
        self.entry_index = entry_index
        self.truth_data = truth_data
        self.use_truth = use_truth
