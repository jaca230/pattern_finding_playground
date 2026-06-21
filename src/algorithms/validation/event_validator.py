from abc import ABC, abstractmethod
from models.event import Event
from typing import Optional, Any

class EventValidator(ABC):
    @abstractmethod
    def validate(self, event: Event, storage: Optional[Any] = None) -> bool:
        """
        Abstract method to validate an event. Should return True if valid, False otherwise.
        """
        pass
