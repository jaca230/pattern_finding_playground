from typing import Any, Optional
import copy
import time

from pipeline.stages.stage import Stage


class Pipeline:
    def __init__(self):
        self.storage = {}
        self._storage_views = {}
        self.registered_stages: dict[str, Stage] = {}
        self.stage_order: list[str] = []
        self.completed_stages: set[str] = set()
        self.current_stage: Optional[str] = None

    def register_stage(self, stage: Stage):
        if stage.stage_key in self.registered_stages:
            raise ValueError(f"Stage '{stage.stage_key}' is already registered")
        self.registered_stages[stage.stage_key] = stage
        self.stage_order.append(stage.stage_key)

    def set_stage_logging(self, enabled: bool = True, stage_key: Optional[str] = None):
        if stage_key is not None:
            if stage_key not in self.registered_stages:
                raise ValueError(f"Stage '{stage_key}' is not registered")
            self.registered_stages[stage_key].logging_enabled = enabled
            return

        for stage in self.registered_stages.values():
            stage.logging_enabled = enabled

    def run(self, input_context: Optional[Any] = None) -> dict:
        self.reset()
        for stage_key in self.stage_order:
            self.execute_stage(stage_key, input_context)
        return self.storage

    def run_next_stage(self, input_context: Optional[Any] = None):
        next_stage = self.get_next_stage()
        if next_stage is None:
            raise RuntimeError("All stages have been completed.")
        self.execute_stage(next_stage, input_context)
        return self.storage

    def run_stage(self, target_stage: str, input_context: Optional[Any] = None):
        self.execute_stage(target_stage, input_context)
        return self.storage

    def go_to_stage(self, target_stage: str, input_context: Optional[Any] = None):
        if target_stage not in self.registered_stages:
            raise ValueError(f"Stage '{target_stage}' is not registered")

        target_index = self.stage_order.index(target_stage)
        for stage_key in self.stage_order[:target_index + 1]:
            if stage_key not in self.completed_stages:
                self.execute_stage(stage_key, input_context)
        return self.storage

    def reset_to_stage(self, target_stage: str, input_context: Optional[Any] = None):
        self.reset()
        return self.go_to_stage(target_stage, input_context)

    def execute_stage(self, stage_key: str, input_context: Optional[Any] = None):
        if not self.can_advance_to_stage(stage_key):
            missing_stages, missing_storage = self.missing_requirements(stage_key)
            error_msg = f"Cannot execute stage '{stage_key}'."
            if missing_stages:
                error_msg += f" Incomplete prerequisite stages: {sorted(missing_stages)}."
            if missing_storage:
                error_msg += f" Missing storage keys: {sorted(missing_storage)}."
            raise RuntimeError(error_msg)

        stage = self.registered_stages[stage_key]
        stage.execute(self.storage, input_context)
        self.current_stage = stage_key
        self.completed_stages.add(stage_key)
        self._storage_views.clear()

    def can_advance_to_stage(self, stage_key: str) -> bool:
        if stage_key not in self.registered_stages:
            return False

        missing_stages, missing_storage = self.missing_requirements(stage_key)
        return not missing_stages and not missing_storage

    def missing_requirements(self, stage_key: str) -> tuple[set[str], set[str]]:
        if stage_key not in self.registered_stages:
            raise ValueError(f"Stage '{stage_key}' is not registered")

        missing_stages = set()
        missing_storage = set()
        stage = self.registered_stages[stage_key]

        for prerequisite in stage.prerequisites:
            if prerequisite in self.registered_stages:
                if prerequisite not in self.completed_stages:
                    missing_stages.add(prerequisite)
            elif prerequisite not in self.storage:
                missing_storage.add(prerequisite)

        return missing_stages, missing_storage

    def get_runnable_stages(self):
        return [
            stage_key
            for stage_key in self.stage_order
            if stage_key not in self.completed_stages and self.can_advance_to_stage(stage_key)
        ]

    def reset(self):
        self.clear_storage()
        self.completed_stages.clear()
        self.current_stage = None

    def clear_storage(self):
        self.storage.clear()
        self._storage_views.clear()

    def get_storage_view(self, keys: list[str]) -> dict[str, Any]:
        key_tuple = tuple(sorted(keys))
        if key_tuple not in self._storage_views:
            self._storage_views[key_tuple] = {k: self.storage.get(k) for k in keys}
        return self._storage_views[key_tuple]

    def get_current_stage(self):
        return self.current_stage

    def get_registered_stages(self):
        return list(self.stage_order)

    def get_completed_stages(self):
        return self.completed_stages.copy()

    def is_stage_completed(self, stage_key: str):
        return stage_key in self.completed_stages

    def get_next_stage(self):
        for stage_key in self.stage_order:
            if stage_key not in self.completed_stages:
                return stage_key
        return None

    def get_storage(self) -> dict:
        return self.storage

    def get_event(self):
        event = self.storage.get("event", None)
        if event is None:
            return None
        return copy.deepcopy(event)

    def get_stage_info(self, stage_key: str) -> dict[str, Any]:
        if stage_key not in self.registered_stages:
            raise ValueError(f"Stage '{stage_key}' is not registered")

        stage = self.registered_stages[stage_key]
        return {
            "name": stage.name,
            "stage_key": stage.stage_key,
            "prerequisites": stage.prerequisites,
            "is_input_stage": stage.is_input_stage,
            "is_completed": self.is_stage_completed(stage_key),
            "is_current": self.current_stage == stage_key,
        }

    def get_pipeline_status(self) -> dict[str, Any]:
        next_stage = self.get_next_stage()
        stage_details = []

        for stage_key in self.stage_order:
            info = self.get_stage_info(stage_key)
            can_run = self.can_advance_to_stage(stage_key)
            info["can_run"] = can_run

            if not can_run:
                missing_stages, missing_storage = self.missing_requirements(stage_key)
                if missing_stages:
                    info["missing_prerequisite_stages"] = sorted(missing_stages)
                if missing_storage:
                    info["missing_storage_keys"] = sorted(missing_storage)

            stage_details.append(info)

        return {
            "total_stages": len(self.stage_order),
            "completed_count": len(self.completed_stages),
            "current_stage": self.current_stage,
            "next_stage": next_stage,
            "all_completed": len(self.completed_stages) == len(self.stage_order),
            "stage_details": stage_details,
        }

    def status(self):
        start_time = time.time()
        status = self.get_pipeline_status()
        status_time = time.time() - start_time

        print(f"Pipeline Status (generated in {status_time:.4f}s):")
        print(f"  Total stages: {status['total_stages']}")
        print(f"  Completed: {status['completed_count']}")
        print(f"  Current stage: {status['current_stage']}")
        print(f"  Next stage: {status['next_stage']}")
        print(f"  All completed: {status['all_completed']}")
        print(f"  Storage keys: {list(self.storage.keys())}")

        print("\nStage Details:")
        for stage_info in status["stage_details"]:
            status_text = "done" if stage_info["is_completed"] else "pending"
            current_text = "current" if stage_info["is_current"] else ""
            runnable_text = "runnable" if stage_info["can_run"] else "blocked"

            print(f"  {stage_info['name']}: {status_text} {current_text} {runnable_text}".strip())
            if not stage_info["can_run"]:
                if "missing_prerequisite_stages" in stage_info:
                    print(f"      Missing stages: {stage_info['missing_prerequisite_stages']}")
                if "missing_storage_keys" in stage_info:
                    print(f"      Missing storage: {stage_info['missing_storage_keys']}")

    def debug_stage(self, stage_key: str):
        try:
            info = self.get_stage_info(stage_key)
            storage_keys = list(self.storage.keys())
            can_run = self.can_advance_to_stage(stage_key)
            missing_stages, missing_storage = self.missing_requirements(stage_key)

            print(f"Stage Debug: {info['name']}")
            print(f"   Stage Key: {info['stage_key']}")
            print(f"   Prerequisites: {info['prerequisites']}")
            print(f"   Is Input Stage: {info['is_input_stage']}")
            print(f"   Is Completed: {info['is_completed']}")
            print(f"   Is Current: {info['is_current']}")
            print(f"   Can Run: {can_run}")
            print(f"   Storage Keys: {storage_keys}")
            print(f"   Completed Stages: {sorted(self.completed_stages)}")

            if missing_stages:
                print(f"   Missing Prerequisite Stages: {sorted(missing_stages)}")
            if missing_storage:
                print(f"   Missing Storage Keys: {sorted(missing_storage)}")
        except ValueError as e:
            print(e)
