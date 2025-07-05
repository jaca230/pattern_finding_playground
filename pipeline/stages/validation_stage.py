from pipeline.staging.stage import Stage
from algorithms.validation.event_validator import EventValidator
from enums.default_stages import DefaultStages
from pipeline.staging.stage import monitor_stage_performance

class ValidationStage(Stage):
    def __init__(self, validator: EventValidator):
        super().__init__(
            name="Validate Event",
            stage_value=DefaultStages.VALIDATION,
            prerequisites=[DefaultStages.EVENT_INIT]
        )
        self.validator = validator

    def build_handler(self):
        @monitor_stage_performance(stage_name=self.name)
        def handler(storage):
            # Direct reference - no copying
            event = storage["event"]
            
            # Call validator and directly assign result
            event.is_valid = self.validator.validate(event, storage=storage)
        return handler