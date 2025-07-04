from typing import List, Union, Optional, Callable, Any
from abc import ABC, abstractmethod

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

    def __repr__(self):
        return f"Stage(name='{self.name}', stage_value={self.stage_value})"

    @abstractmethod
    def build_handler(self) -> Callable[[Any], None]:
        """Return the callable to execute for this stage"""
        pass
