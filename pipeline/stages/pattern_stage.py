from pipeline.staging.stage import Stage
from algorithms.pattern.pattern_former import PatternFormer
from enums.default_stages import DefaultStages
from pipeline.staging.stage import monitor_stage_performance

class PatternStage(Stage):
    def __init__(self, pattern_former: PatternFormer):
        super().__init__(
            name="Form Patterns",
            stage_value=DefaultStages.PATTERNS,
            prerequisites=[DefaultStages.VERTICES]
        )
        self.pattern_former = pattern_former

    def build_handler(self):
        @monitor_stage_performance(stage_name=self.name)
        def handler(storage):
            # Direct reference - no copying
            vertices = storage["vertices"]
            
            # Call the algorithm
            patterns, info = self.pattern_former.form_patterns(vertices, storage=storage)

            # Optimize set operations - check if patterns exist first
            if "patterns" not in storage:
                storage["patterns"] = set(patterns)  # Create with initial data
            else:
                # Clear and update in one operation to maintain references
                existing_patterns = storage["patterns"]
                existing_patterns.clear()
                existing_patterns.update(patterns)
            
            # Use setdefault to avoid double lookup
            storage.setdefault("extra_info", {})["pattern_algorithm_info"] = info
        return handler