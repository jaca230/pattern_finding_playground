
from typing import Any, Optional, Dict, List, Union, Set, Callable
from abc import ABC, abstractmethod
import time
from collections import defaultdict
from config.config import ENABLE_STAGE_LOGGING

def monitor_stage_performance(stage_name=None):
    """Decorator to monitor stage execution time, if enabled"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if ENABLE_STAGE_LOGGING:
                import time
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                elapsed = end_time - start_time
                print(f"Stage '{stage_name or func.__name__}' executed in {elapsed:.4f}s")
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator



class Stage(ABC):
    def __init__(
        self, 
        name: str, 
        stage_value: Union[int, object], 
        prerequisites: Optional[List[Union[int, object]]] = None,
        is_input_stage: bool = False
    ):
        self.name = name
        self.stage_value = stage_value
        self.prerequisites = prerequisites or []
        self.is_input_stage = is_input_stage
        # Cache the handler to avoid repeated calls to build_handler
        self._handler = None

    def __repr__(self):
        return f"Stage(name='{self.name}', stage_value={self.stage_value})"

    @abstractmethod
    def build_handler(self) -> Callable[[Any], None]:
        """Return the callable to execute for this stage"""
        pass
    
    def get_handler(self) -> Callable[[Any], None]:
        """Get cached handler or build it once"""
        if self._handler is None:
            self._handler = self.build_handler()
        return self._handler