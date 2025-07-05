from typing import Any, Optional, List, Dict
from pipeline.staging.stage_manager import StageManager
from pipeline.staging.stage import Stage
from collections import defaultdict

class Pipeline:
    def __init__(self):
        self.stage_manager = StageManager()
        self.storage = {}
        self._storage_views = {}  # Cache for storage views
        
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
        """Run a specific stage"""
        if target_stage not in self.stage_manager.registered_steps:
            raise ValueError(f"Stage {target_stage} not registered in pipeline")
        
        self.stage_manager.execute_stage(target_stage, self.storage, input_context)
        return self.storage

    def go_to_stage(self, target_stage, input_context: Optional[Any] = None):
        """Go to a specific stage, running prerequisites if needed"""
        self.stage_manager.run_to_stage(target_stage, self.storage, input_context)
        return self.storage

    def reset_to_stage(self, target_stage, input_context: Optional[Any] = None):
        """Reset and run up to target stage"""
        self.reset()
        return self.go_to_stage(target_stage, input_context)

    def get_runnable_stages(self):
        """Get stages that can run right now"""
        return self.stage_manager.get_runnable_stages(self.storage)

    def reset(self):
        """Reset the pipeline"""
        self.clear_storage()
        self.stage_manager.reset()
        self._storage_views.clear()

    def clear_storage(self):
        self.storage.clear()
        self._storage_views.clear()

    def get_storage_view(self, keys: List[str]) -> Dict[str, Any]:
        """Get a cached view of specific storage keys"""
        key_tuple = tuple(sorted(keys))
        if key_tuple not in self._storage_views:
            self._storage_views[key_tuple] = {k: self.storage.get(k) for k in keys}
        return self._storage_views[key_tuple]

    # Keep all existing methods for compatibility
    def get_current_stage(self):
        return self.stage_manager.current_stage

    def get_registered_stages(self):
        return self.stage_manager.get_sorted_stages()

    def get_completed_stages(self):
        return self.stage_manager.get_completed_stages()

    def is_stage_completed(self, stage_value):
        return self.stage_manager.is_stage_completed(stage_value)

    def get_next_stage(self):
        return self.stage_manager.get_next_uncompleted_stage()

    def get_storage(self) -> dict:
        return self.storage
    
    def get_event(self):
        event = self.storage.get("event", None)
        if event is None:
            return None
        # Return a deep copy for safety
        import copy
        return copy.deepcopy(event)

    def status(self):
        """Get pipeline status with timing info"""
        start_time = time.time()
        status = self.stage_manager.get_pipeline_status(self.storage)
        status_time = time.time() - start_time
        
        print(f"Pipeline Status (generated in {status_time:.4f}s):")
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
                # Use optimized dependency checking
                missing_stages = (self.stage_manager._prerequisite_graph.get(stage_value, set()) - 
                                self.stage_manager.completed_stages)
                missing_storage = (self.stage_manager._storage_dependencies.get(stage_value, set()) - 
                                 set(storage_keys))
                
                if missing_stages:
                    print(f"   Missing Prerequisite Stages: {list(missing_stages)}")
                if missing_storage:
                    print(f"   Missing Storage Keys: {list(missing_storage)}")
        except ValueError as e:
            print(f"❌ {e}")