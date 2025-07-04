from typing import Any, Optional
from pipeline.staging.stage_manager import StageManager
from pipeline.staging.stage import Stage
import copy

class Pipeline:
    def __init__(self):
        self.stage_manager = StageManager()
        self.storage = {}

    def register_stage(self, stage: Stage):
        self.stage_manager.register_step(stage)

    def run(self, input_context: Optional[Any] = None) -> dict:
        """Run all stages from the beginning"""
        self.reset()
        self.stage_manager.run_all_stages(self.storage, input_context)
        return self.storage

    def run_next_stage(self, input_context: Optional[Any] = None):
        """Run the next uncompleted stage"""
        next_stage = self.stage_manager.get_next_uncompleted_stage()
        if next_stage is None:
            raise RuntimeError("All stages have been completed.")
        
        self.stage_manager.execute_stage(next_stage, self.storage, input_context)
        return self.storage

    def run_stage(self, target_stage, input_context: Optional[Any] = None):
        """
        Run a specific stage. Perfect for re-running individual Jupyter cells.
        Marks the stage as completed when successful.
        """
        if target_stage not in self.stage_manager.registered_steps:
            raise ValueError(f"Stage {target_stage} not registered in pipeline")
        
        self.stage_manager.execute_stage(target_stage, self.storage, input_context)
        return self.storage

    def go_to_stage(self, target_stage, input_context: Optional[Any] = None):
        """
        Go to a specific stage, running all prerequisite stages if needed.
        Only runs uncompleted stages.
        """
        self.stage_manager.run_to_stage(target_stage, self.storage, input_context)
        return self.storage

    def reset_to_stage(self, target_stage, input_context: Optional[Any] = None):
        """
        Reset the pipeline and run up to the target stage.
        This clears storage and starts fresh.
        """
        self.reset()
        return self.go_to_stage(target_stage, input_context)

    def reset(self):
        """Reset the pipeline to initial state"""
        self.clear_storage()
        self.stage_manager.reset()

    def get_current_stage(self):
        """Get the current stage the pipeline is at"""
        return self.stage_manager.current_stage

    def get_registered_stages(self):
        """Get all registered stages in order"""
        return self.stage_manager.get_sorted_stages()

    def get_completed_stages(self):
        """Get all completed stages"""
        return self.stage_manager.get_completed_stages()

    def is_stage_completed(self, stage_value):
        """Check if a stage has been completed"""
        return self.stage_manager.is_stage_completed(stage_value)

    def get_next_stage(self):
        """Get the next uncompleted stage, or None if all are completed"""
        return self.stage_manager.get_next_uncompleted_stage()

    def clear_storage(self):
        self.storage.clear()

    def get_storage(self) -> dict:
        return self.storage
    
    def get_event(self):
        event = self.storage.get("event", None)
        if event is None:
            return None
        # Return a deep copy so the user can safely modify it without affecting storage
        return copy.deepcopy(event)

    def status(self):
        """Get a summary of pipeline status"""
        status = self.stage_manager.get_pipeline_status(self.storage)
        
        print(f"Pipeline Status:")
        print(f"  Total stages: {status['total_stages']}")
        print(f"  Completed: {status['completed_count']}")
        print(f"  Current stage: {status['current_stage']}")
        print(f"  Next stage: {status['next_stage']}")
        print(f"  All completed: {status['all_completed']}")
        print(f"  Storage keys: {list(self.storage.keys())}")
        
        print(f"\nStage Details:")
        for stage_info in status['stage_details']:
            status_icon = "✅" if stage_info['is_completed'] else "⏳"
            current_icon = "👉" if stage_info['is_current'] else "  "
            can_run_icon = "🚀" if stage_info['can_run'] else "❌"
            
            print(f"  {status_icon} {current_icon} {stage_info['name']} {can_run_icon}")
            if not stage_info['can_run']:
                if 'missing_prerequisite_stages' in stage_info:
                    print(f"      Missing stages: {stage_info['missing_prerequisite_stages']}")
                if 'missing_storage_keys' in stage_info:
                    print(f"      Missing storage: {stage_info['missing_storage_keys']}")

    def debug_stage(self, stage_value):
        """Debug information for a specific stage"""
        try:
            info = self.stage_manager.get_stage_info(stage_value)
            storage_keys = list(self.storage.keys())
            can_run = self.stage_manager.can_advance_to_stage(stage_value, self.storage)
            
            print(f"🔍 Stage Debug: {info['name']}")
            print(f"   Stage Value: {info['stage_value']}")
            print(f"   Prerequisites: {info['prerequisites']}")
            print(f"   Is Input Stage: {info['is_input_stage']}")
            print(f"   Is Completed: {info['is_completed']}")
            print(f"   Is Current: {info['is_current']}")
            print(f"   Can Run: {can_run}")
            print(f"   Storage Keys: {storage_keys}")
            print(f"   Completed Stages: {list(self.stage_manager.completed_stages)}")
            
            if not can_run:
                stage = self.stage_manager.registered_steps[stage_value]
                missing_stages = []
                missing_storage = []
                
                for prereq in stage.prerequisites:
                    if isinstance(prereq, (int, object)) and prereq in self.stage_manager.registered_steps:
                        # This is a stage prerequisite
                        if prereq not in self.stage_manager.completed_stages:
                            missing_stages.append(prereq)
                    else:
                        # This is a storage key prerequisite
                        if prereq not in storage_keys:
                            missing_storage.append(prereq)
                
                if missing_stages:
                    print(f"   Missing Prerequisite Stages: {missing_stages}")
                if missing_storage:
                    print(f"   Missing Storage Keys: {missing_storage}")
        except ValueError as e:
            print(f"❌ {e}")