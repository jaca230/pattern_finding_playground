from abc import ABC, abstractmethod
from typing import List, Any
from models.tracklet import Tracklet

class TrackletFormer(ABC):
    @abstractmethod
    def form_tracklets(self, file: Any, entry_index: int) -> List[Tracklet]:
        """
        Forms a list of Tracklet objects from the given file and entry index.

        Args:
            file: A ROOT file or similar object that provides access to event data.
            entry_index: The index of the event within the file.

        Returns:
            A list of Tracklet objects for the given event.
        """
        pass
