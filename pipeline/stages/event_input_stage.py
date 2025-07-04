# pipeline/stages/input_stage.py

from pipeline.staging.stage import Stage
from enums.default_stages import DefaultStages

class EventInputStage(Stage):
    def __init__(self):
        super().__init__(
            name="Load Event Input Data",
            stage_value=DefaultStages.INPUT,
            prerequisites=[],
            is_input_stage=True
        )

    def build_handler(self):
        # handler accepts storage and input_context
        def handler(storage, input_context):
            storage["tree"] = input_context.tree
            storage["geo"] = input_context.geo
            storage["entry_index"] = input_context.entry_index
        return handler

# Simple class to hold inputs for EventInputStage runs
class InputContext:
    def __init__(self, tree, geo, entry_index):
        self.tree = tree
        self.geo = geo
        self.entry_index = entry_index
