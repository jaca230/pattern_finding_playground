from abc import ABC, abstractmethod
from typing import List, Any, Optional
from models.tracklet import Tracklet

class TrackletFormer(ABC):
    @abstractmethod
    def form_tracklets(
        self,
        reco_event: Any,
        geoHelper: Any,
        storage: Optional[Any] = None,
        truth_event: Optional[Any] = None,
    ) -> tuple[List[Tracklet], dict]:
        """
        Forms a list of Tracklet objects from a loaded Reco RNTuple event.

        Args:
            reco_event: Loaded event from the reconstructed RNTuple collections.
            geoHelper: A ROOT helper object (e.g., PIMCGeoHelper) for geometry lookups.
            truth_event: Optional loaded event from the truth-guided RNTuple collections.

        Returns:
            A list of Tracklet objects for the given event.
        """
        pass
