from pipeline.stages.stage import Stage

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
            storage["reco_data"] = input_context.reco_data
            storage["truth_data"] = input_context.truth_data
            storage["geo"] = input_context.reco_data.geo
            storage["entry_index"] = input_context.entry_index
            storage["reco_entry"] = input_context.reco_data.load_entry(input_context.entry_index)
            storage["truth_entry"] = (
                input_context.truth_data.load_entry(input_context.entry_index)
                if input_context.truth_data is not None
                else None
            )
        return handler

# Simple class to hold inputs for EventInputStage runs
class InputContext:
    __slots__ = ("reco_data", "entry_index", "truth_data")

    def __init__(self, reco_data, entry_index, truth_data=None):
        self.reco_data = reco_data
        self.entry_index = entry_index
        self.truth_data = truth_data
