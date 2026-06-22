from abc import ABC, abstractmethod
from typing import List, Any, Optional
from models.tracklet import Tracklet

class TrackletFormer(ABC):
    def _root_tracklet_hits(self, root_tracklet: Any, hit_vector: Any) -> list[Any]:
        """Return ROOT hit objects referenced by a ROOT tracklet's hit-index list."""
        hits = []
        for hit_index in root_tracklet.GetAtarHitIndices():
            hit_index = int(hit_index)
            if 0 <= hit_index < hit_vector.size():
                hits.append(hit_vector[hit_index])
        return hits

    @abstractmethod
    def form_tracklets(
        self,
        reco_entry: Any,
        geo: Any,
        storage: Optional[Any] = None,
        truth_entry: Optional[Any] = None,
    ) -> tuple[List[Tracklet], dict]:
        """
        Forms a list of Tracklet objects from loaded Reco RNTuple collections.

        Args:
            reco_entry: Dict with loaded reconstructed `patterns`, `tracklets`, and `hits`.
            geo: The `GeoHeader` object from the same ROOT file, used for geometry lookups.
            truth_entry: Optional dict with loaded truth-guided collections.

        Returns:
            A list of Tracklet objects for the given event.
        """
        pass
