from pipeline.staging.stage import Stage
from models.event import Event
from enums.default_stages import DefaultStages
from pipeline.staging.stage import monitor_stage_performance

class EventInitStage(Stage):
    def __init__(self):
        super().__init__(
            name="Initialize Event",
            stage_value=DefaultStages.EVENT_INIT,
            prerequisites=[DefaultStages.INPUT]
        )

    def build_handler(self):
        @monitor_stage_performance(stage_name=self.name)
        def handler(storage):
            entry_index = storage["entry_index"]
            event = Event(event_id=entry_index)

            # Pre-initialize collections to avoid repeated key lookups
            if "patterns" not in storage:
                storage["patterns"] = set()
            if "extra_info" not in storage:
                storage["extra_info"] = {}
            if "tracklets" not in storage:
                storage["tracklets"] = set()

            # Direct reference assignment - no copying
            event.set_patterns(storage["patterns"])
            event.extra_info = storage["extra_info"]
            event.all_tracklets = storage["tracklets"]

            storage["event"] = event
        return handler