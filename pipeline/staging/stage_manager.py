from typing import Callable, Dict, Union, Optional, Any, Set, List
from pipeline.staging.stage import Stage
from collections import defaultdict

class StageManager:
    def __init__(self):
        self.current_stage: Optional[Union[int, object]] = None
        self.registered_steps: Dict[Union[int, object], Stage] = {}
        self.completed_stages: Set[Union[int, object]] = set()
        
        # Performance optimizations
        self._sorted_stages_cache: Optional[List[Union[int, object]]] = None
        self._prerequisite_graph: Dict[Union[int, object], Set[Union[int, object]]] = {}
        self._storage_dependencies: Dict[Union[int, object], Set[str]] = {}
        self._stage_to_dependents: Dict[Union[int, object], Set[Union[int, object]]] = defaultdict(set)
        
    def register_step(self, stage: Stage):
        """Register a stage and build dependency graph"""
        self.registered_steps[stage.stage_value] = stage
        
        # Invalidate cache
        self._sorted_stages_cache = None
        
        # Build dependency graphs for faster lookups
        stage_prereqs = set()
        storage_prereqs = set()
        
        for prereq in stage.prerequisites:
            if isinstance(prereq, (int, object)) and prereq in self.registered_steps:
                stage_prereqs.add(prereq)
                self._stage_to_dependents[prereq].add(stage.stage_value)
            else:
                storage_prereqs.add(prereq)
        
        self._prerequisite_graph[stage.stage_value] = stage_prereqs
        self._storage_dependencies[stage.stage_value] = storage_prereqs

    def _get_sorted_stages(self) -> List[Union[int, object]]:
        """Get cached sorted stages"""
        if self._sorted_stages_cache is None:
            self._sorted_stages_cache = sorted(
                self.registered_steps.keys(),
                key=lambda sv: getattr(sv, 'value', sv)
            )
        return self._sorted_stages_cache

    def can_advance_to_stage(self, target_stage: Union[int, object], storage: Dict) -> bool:
        """Optimized prerequisite checking using cached dependency graphs"""
        if target_stage not in self.registered_steps:
            return False
        
        # Check stage prerequisites using cached graph
        stage_prereqs = self._prerequisite_graph.get(target_stage, set())
        if not stage_prereqs.issubset(self.completed_stages):
            return False
        
        # Check storage prerequisites using cached dependencies
        storage_prereqs = self._storage_dependencies.get(target_stage, set())
        storage_keys = storage.keys()
        if not storage_prereqs.issubset(storage_keys):
            return False
        
        return True

    def execute_stage(
        self, 
        target_stage: Union[int, object], 
        storage: Dict, 
        input_context: Optional[Any] = None
    ):
        """Execute a stage with optimized error handling"""
        if not self.can_advance_to_stage(target_stage, storage):
            # Use cached dependency info for faster error reporting
            missing_stages = self._prerequisite_graph[target_stage] - self.completed_stages
            missing_storage = self._storage_dependencies[target_stage] - storage.keys()
            
            error_msg = f"Cannot execute stage {target_stage}."
            if missing_stages:
                error_msg += f" Incomplete prerequisite stages: {missing_stages}"
            if missing_storage:
                error_msg += f" Missing storage keys: {missing_storage}"
            
            raise RuntimeError(error_msg)

        stage = self.registered_steps[target_stage]
        handler = stage.get_handler()  # Use cached handler
        
        # Pass input_context only if this is an input stage
        if stage.is_input_stage:
            handler(storage, input_context)
        else:
            handler(storage)

        self.current_stage = target_stage
        self.completed_stages.add(target_stage)

    def get_next_uncompleted_stage(self) -> Optional[Union[int, object]]:
        """Get the next uncompleted stage using cached sorted list"""
        sorted_stages = self._get_sorted_stages()
        for stage_value in sorted_stages:
            if stage_value not in self.completed_stages:
                return stage_value
        return None

    def batch_execute_stages(
        self, 
        stage_list: List[Union[int, object]], 
        storage: Dict, 
        input_context: Optional[Any] = None
    ):
        """Execute multiple stages in batch for better performance"""
        for stage_value in stage_list:
            if stage_value not in self.completed_stages:
                self.execute_stage(stage_value, storage, input_context)

    def run_all_stages(self, storage: Dict, input_context: Optional[Any] = None):
        """Optimized run all stages"""
        sorted_stages = self._get_sorted_stages()
        self.batch_execute_stages(sorted_stages, storage, input_context)

    def run_to_stage(self, target_stage: Union[int, object], storage: Dict, input_context: Optional[Any] = None):
        """Optimized run to stage"""
        if target_stage not in self.registered_steps:
            raise ValueError(f"Stage {target_stage} not registered")
        
        sorted_stages = self._get_sorted_stages()
        target_index = sorted_stages.index(target_stage)
        
        # Get stages to run in one slice operation
        stages_to_run = sorted_stages[:target_index + 1]
        self.batch_execute_stages(stages_to_run, storage, input_context)

    def get_runnable_stages(self, storage: Dict) -> List[Union[int, object]]:
        """Get all stages that can currently run (useful for parallel execution)"""
        runnable = []
        for stage_value in self.registered_steps.keys():
            if (stage_value not in self.completed_stages and 
                self.can_advance_to_stage(stage_value, storage)):
                runnable.append(stage_value)
        return runnable

    def reset(self):
        """Reset the stage manager"""
        self.current_stage = None
        self.completed_stages.clear()

    # Keep existing methods for compatibility
    def get_sorted_stages(self) -> List[Union[int, object]]:
        return self._get_sorted_stages()

    def is_stage_completed(self, stage_value: Union[int, object]) -> bool:
        return stage_value in self.completed_stages

    def get_completed_stages(self) -> Set[Union[int, object]]:
        return self.completed_stages.copy()

    def get_stage_info(self, stage_value: Union[int, object]) -> Dict[str, Any]:
        """Get stage info using cached dependency data"""
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
        """Optimized pipeline status"""
        sorted_stages = self._get_sorted_stages()
        next_stage = self.get_next_uncompleted_stage()
        
        stage_details = []
        storage_keys = storage.keys()  # Cache keys lookup
        
        for stage_value in sorted_stages:
            info = self.get_stage_info(stage_value)
            can_run = self.can_advance_to_stage(stage_value, storage)
            info['can_run'] = can_run
            
            if not can_run:
                # Use cached dependency info
                missing_stages = self._prerequisite_graph[stage_value] - self.completed_stages
                missing_storage = self._storage_dependencies[stage_value] - storage_keys
                
                if missing_stages:
                    info['missing_prerequisite_stages'] = list(missing_stages)
                if missing_storage:
                    info['missing_storage_keys'] = list(missing_storage)
            
            stage_details.append(info)
        
        return {
            'total_stages': len(sorted_stages),
            'completed_count': len(self.completed_stages),
            'current_stage': self.current_stage,
            'next_stage': next_stage,
            'all_completed': len(self.completed_stages) == len(sorted_stages),
            'stage_details': stage_details
        }
