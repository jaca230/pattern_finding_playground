from abc import ABC, abstractmethod
from typing import List, Any
from models.tracklet import Tracklet

class TrackletFormer(ABC):
    @abstractmethod
    def form_tracklets(self, tree: Any, geoHelper: Any, entry_index: int) -> List[Tracklet]:
        """
        Forms a list of Tracklet objects from the given tree and geoHelper at a specified entry.

        Args:
            tree: The ROOT TTree containing the event data.
            geoHelper: A ROOT helper object (e.g., PIMCGeoHelper) for geometry lookups.
            entry_index: The index of the event within the tree.

        Returns:
            A list of Tracklet objects for the given event.
        """
        pass
