from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
import time


class Stage(ABC):
    def __init__(
        self,
        name: str,
        stage_key: str,
        prerequisites: Optional[list[str]] = None,
        is_input_stage: bool = False,
        logging_enabled: bool = False,
    ):
        self.name = name
        self.stage_key = stage_key
        self.prerequisites = prerequisites or []
        self.is_input_stage = is_input_stage
        self.logging_enabled = logging_enabled
        self._handler = None

    def __repr__(self):
        return f"Stage(name='{self.name}', stage_key='{self.stage_key}')"

    @abstractmethod
    def build_handler(self) -> Callable[[Any], None]:
        """Return the callable that executes this stage."""
        pass

    def get_handler(self) -> Callable[[Any], None]:
        if self._handler is None:
            self._handler = self.build_handler()
        return self._handler

    def execute(self, storage: dict, input_context: Optional[Any] = None):
        handler = self.get_handler()
        start_time = time.time()

        if self.is_input_stage:
            handler(storage, input_context)
        else:
            handler(storage)

        if self.logging_enabled:
            elapsed = time.time() - start_time
            print(f"Stage '{self.name}' executed in {elapsed:.4f}s")
