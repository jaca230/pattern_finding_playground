from pipeline.staging.stage import Stage
from models.event import Event
from enums.default_stages import DefaultStages

class EventInitStage(Stage):
    def __init__(self):
        super().__init__(
            name="Initialize Event",
            stage_value=DefaultStages.EVENT_INIT,
            prerequisites=[DefaultStages.INPUT]
        )

    def build_handler(self):
        def handler(storage):
            entry_index = storage["entry_index"]
            event = Event(event_id=entry_index)

            if "patterns" not in storage:
                storage["patterns"] = set()

            event.set_patterns(storage["patterns"])

            if "extra_info" not in storage:
                storage["extra_info"] = {}
            if "tracklets" not in storage:
                storage["tracklets"] = set()

            # Link event.extra_info to the shared extra_info dict in storage
            event.extra_info = storage["extra_info"]
            # Ensure tracklets are set in the event
            event.all_tracklets = storage["tracklets"]

            storage["event"] = event
        return handler
