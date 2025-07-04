from typing import Callable, Dict, Union, Optional, Any, Set, List
from pipeline.staging.stage import Stage

class StageManager:
    def __init__(self):
        self.current_stage: Optional[Union[int, object]] = None
        self.registered_steps: Dict[Union[int, object], Stage] = {}
        self.step_handlers: Dict[Union[int, object], Callable] = {}
        self.completed_stages: Set[Union[int, object]] = set()

    def register_step(self, stage: Stage):
        self.registered_steps[stage.stage_value] = stage
        self.step_handlers[stage.stage_value] = stage.build_handler()

    def can_advance_to_stage(self, target_stage: Union[int, object], storage: Dict) -> bool:
        if target_stage not in self.registered_steps:
            return False
        stage = self.registered_steps[target_stage]
        
        # Check if all prerequisite stages have been completed
        for prereq in stage.prerequisites:
            if isinstance(prereq, (int, object)) and prereq in self.registered_steps:
                # This is a stage prerequisite - check if it's completed
                if prereq not in self.completed_stages:
                    return False
            else:
                # This is a storage key prerequisite - check if it exists in storage
                if prereq not in storage:
                    return False
        
        return True

    def execute_stage(
        self, 
        target_stage: Union[int, object], 
        storage: Dict, 
        input_context: Optional[Any] = None
    ):
        """Execute a stage and mark it as completed"""
        if not self.can_advance_to_stage(target_stage, storage):
            stage = self.registered_steps[target_stage]
            missing_stages = []
            missing_storage = []
            
            for prereq in stage.prerequisites:
                if isinstance(prereq, (int, object)) and prereq in self.registered_steps:
                    # This is a stage prerequisite
                    if prereq not in self.completed_stages:
                        missing_stages.append(prereq)
                else:
                    # This is a storage key prerequisite
                    if prereq not in storage:
                        missing_storage.append(prereq)
            
            error_msg = f"Cannot execute stage {target_stage}."
            if missing_stages:
                error_msg += f" Incomplete prerequisite stages: {missing_stages}"
            if missing_storage:
                error_msg += f" Missing storage keys: {missing_storage}"
            
            raise RuntimeError(error_msg)

        stage = self.registered_steps[target_stage]
        handler = self.step_handlers[target_stage]
        
        # Pass input_context only if this is an input stage
        if stage.is_input_stage:
            handler(storage, input_context)
        else:
            handler(storage)

        self.current_stage = target_stage
        self.completed_stages.add(target_stage)

    def advance_to_stage(
        self, 
        target_stage: Union[int, object], 
        storage: Dict, 
        input_context: Optional[Any] = None
    ):
        """Legacy method - delegates to execute_stage"""
        self.execute_stage(target_stage, storage, input_context)

    def get_sorted_stages(self) -> List[Union[int, object]]:
        """Get all registered stages in sorted order"""
        return sorted(
            self.registered_steps.keys(),
            key=lambda sv: getattr(sv, 'value', sv)
        )

    def get_next_uncompleted_stage(self) -> Optional[Union[int, object]]:
        """Get the next uncompleted stage, or None if all are completed"""
        sorted_stages = self.get_sorted_stages()
        for stage_value in sorted_stages:
            if stage_value not in self.completed_stages:
                return stage_value
        return None

    def is_stage_completed(self, stage_value: Union[int, object]) -> bool:
        """Check if a stage has been completed"""
        return stage_value in self.completed_stages

    def get_completed_stages(self) -> Set[Union[int, object]]:
        """Get all completed stages"""
        return self.completed_stages.copy()

    def run_all_stages(self, storage: Dict, input_context: Optional[Any] = None):
        """Run all stages in order"""
        sorted_stages = self.get_sorted_stages()
        for stage_value in sorted_stages:
            self.execute_stage(stage_value, storage, input_context)

    def run_to_stage(self, target_stage: Union[int, object], storage: Dict, input_context: Optional[Any] = None):
        """Run all stages up to and including the target stage (only uncompleted ones)"""
        if target_stage not in self.registered_steps:
            raise ValueError(f"Stage {target_stage} not registered")
        
        sorted_stages = self.get_sorted_stages()
        target_index = sorted_stages.index(target_stage)
        
        # Run all stages up to and including the target stage (only if not completed)
        for i in range(target_index + 1):
            stage_value = sorted_stages[i]
            if stage_value not in self.completed_stages:
                self.execute_stage(stage_value, storage, input_context)

    def reset(self):
        """Reset the stage manager to initial state"""
        self.current_stage = None
        self.completed_stages.clear()

    def get_stage_info(self, stage_value: Union[int, object]) -> Dict[str, Any]:
        """Get detailed information about a stage"""
        if stage_value not in self.registered_steps:
            raise ValueError(f"Stage {stage_value} not registered")
        
        stage = self.registered_steps[stage_value]
        return {
            'name': stage.name,
            'stage_value': stage.stage_value,
            'prerequisites': stage.prerequisites,
            'is_input_stage': stage.is_input_stage,
            'is_completed': self.is_stage_completed(stage_value),
            'is_current': self.current_stage == stage_value
        }

    def get_pipeline_status(self, storage: Dict) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        sorted_stages = self.get_sorted_stages()
        next_stage = self.get_next_uncompleted_stage()
        
        stage_details = []
        for stage_value in sorted_stages:
            info = self.get_stage_info(stage_value)
            can_run = self.can_advance_to_stage(stage_value, storage)
            info['can_run'] = can_run
            if not can_run:
                stage = self.registered_steps[stage_value]
                missing_stages = []
                missing_storage = []
                
                for prereq in stage.prerequisites:
                    if isinstance(prereq, (int, object)) and prereq in self.registered_steps:
                        # This is a stage prerequisite
                        if prereq not in self.completed_stages:
                            missing_stages.append(prereq)
                    else:
                        # This is a storage key prerequisite
                        if prereq not in storage:
                            missing_storage.append(prereq)
                
                if missing_stages:
                    info['missing_prerequisite_stages'] = missing_stages
                if missing_storage:
                    info['missing_storage_keys'] = missing_storage
            stage_details.append(info)
        
        return {
            'total_stages': len(sorted_stages),
            'completed_count': len(self.completed_stages),
            'current_stage': self.current_stage,
            'next_stage': next_stage,
            'all_completed': len(self.completed_stages) == len(sorted_stages),
            'stage_details': stage_details
        }