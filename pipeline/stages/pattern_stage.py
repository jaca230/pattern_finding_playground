from pipeline.staging.stage import Stage
from algorithms.pattern.pattern_former import PatternFormer
from enums.default_stages import DefaultStages

class PatternStage(Stage):
    def __init__(self, pattern_former: PatternFormer):
        super().__init__(
            name="Form Patterns",
            stage_value=DefaultStages.PATTERNS,
            prerequisites=[DefaultStages.VERTICES]
        )
        self.pattern_former = pattern_former

    def build_handler(self):
        def handler(storage):
            vertices = storage["vertices"]
            patterns, info = self.pattern_former.form_patterns(vertices, storage=storage)

            # Initialize the set if missing
            if "patterns" not in storage:
                storage["patterns"] = set()
            # Clear and update in place so references remain valid
            storage["patterns"].update(patterns)
            storage.setdefault("extra_info", {})["pattern_algorithm_info"] = info
        return handler

