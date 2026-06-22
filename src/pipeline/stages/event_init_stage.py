from pipeline.stages.stage import Stage
from models.event import Event

class EventInitStage(Stage):
    def __init__(self):
        super().__init__(
            name="Initialize Event",
            stage_key="event_init",
            prerequisites=["input"]
        )

    def build_handler(self):
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
            if "vertices" not in storage:
                storage["vertices"] = set()
            if "raw_hits" not in storage:
                storage["raw_hits"] = []

            # Direct reference assignment - no copying
            event.set_patterns(storage["patterns"])
            event.extra_info = storage["extra_info"]
            event.all_tracklets = storage["tracklets"]
            event.all_vertices = storage["vertices"]
            event.all_hits = storage["raw_hits"]

            storage["event"] = event
        return handler
