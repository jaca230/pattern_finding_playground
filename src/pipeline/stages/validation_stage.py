from pipeline.stages.stage import Stage
from algorithms.validation.event_validator import EventValidator

class ValidationStage(Stage):
    def __init__(self, validator: EventValidator):
        super().__init__(
            name="Validate Event",
            stage_key="validation",
            prerequisites=["event_init"]
        )
        self.validator = validator

    def build_handler(self):
        def handler(storage):
            # Direct reference - no copying
            event = storage["event"]
            
            # Call validator and directly assign result
            event.is_valid = self.validator.validate(event, storage=storage)
        return handler
