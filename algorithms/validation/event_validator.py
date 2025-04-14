from abc import ABC, abstractmethod

class EventValidator(ABC):
    @abstractmethod
    def validate(self, event: 'EventPatterns') -> bool:
        """
        Abstract method to validate an event. Should return True if valid, False otherwise.
        """
        pass
