from pipeline.staging.stage import Stage
from algorithms.validation.event_validator import EventValidator
from enums.default_stages import DefaultStages

class ValidationStage(Stage):
    def __init__(self, validator: EventValidator):
        super().__init__(
            name="Validate Event",
            stage_value=DefaultStages.VALIDATION,
            prerequisites=[DefaultStages.EVENT_INIT]
        )
        self.validator = validator

    def build_handler(self):
        def handler(storage):
            event = storage["event"]
            is_valid = self.validator.validate(event, storage=storage)
            event.is_valid = is_valid
        return handler
