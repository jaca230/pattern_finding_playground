from typing import List, Optional, Callable
from models.hit import Hit
from utils.particle_mapping import particle_name_map  # Make sure this is the correct import
from models.point_3d import Point3D  # Make sure this import is correct


class Tracklet:
    def __init__(
        self,
        tracklet_id: int,
        particle_id: int,
        e_id: int,
        hits: List[Hit],
        fitter: Optional[Callable[[List[Hit]], dict]] = None,
        is_incoming: Optional[bool] = None,
        delta_r: Optional[float] = None,
        endpoint_0: Optional[Point3D] = None,
        endpoint_1: Optional[Point3D] = None
    ):
        self.tracklet_id = tracklet_id
        self.particle_id = particle_id
        self.e_id = e_id
        self.hits = hits
        self.particle_name, self.particle_color = self.get_particle_info(particle_id)
        self.fitter = fitter
        self.is_incoming = is_incoming
        self.delta_r = delta_r
        self.endpoint_0 = endpoint_0
        self.endpoint_1 = endpoint_1
        self.fit_results: Optional[dict] = {}

    def get_particle_info(self, particle_id: int):
        """Retrieves the particle name and color based on the particle ID."""
        particle_info = particle_name_map.get(particle_id, particle_name_map['default'])
        return particle_info['name'], particle_info['color']

    def get_front_hits(self) -> List[Hit]:
        """Returns the hits from the front detector side."""
        return [hit for hit in self.hits if hit.detector_side == 'front']

    def get_back_hits(self) -> List[Hit]:
        """Returns the hits from the back detector side."""
        return [hit for hit in self.hits if hit.detector_side == 'back']

    def get_endpoints(self) -> tuple[Optional[Point3D], Optional[Point3D]]:
        """Returns the two endpoints as Point3D objects."""
        return self.endpoint_0, self.endpoint_1

    def set_endpoints(self, point_0: Point3D, point_1: Point3D):
        """Sets the two endpoints (Point3D objects)."""
        self.endpoint_0 = point_0
        self.endpoint_1 = point_1

    def fit(self) -> dict:
        """Applies the fitting function to the hits and stores the results."""
        if not self.fitter:
            raise RuntimeError("No fitter function provided.")
        self.fit_results = self.fitter(self.hits)
        return self.fit_results

    def __repr__(self) -> str:
        endpoint_repr = (
            f", endpoints=({repr(self.endpoint_0)}, {repr(self.endpoint_1)})"
            if self.endpoint_0 and self.endpoint_1 else ""
        )
        return (
            f"Tracklet(id={self.tracklet_id}, particle_id={self.particle_id}, "
            f"name={self.particle_name}, color={self.particle_color}, "
            f"e_id={self.e_id}, hits={len(self.hits)}, "
            f"is_incoming={self.is_incoming}, delta_r={self.delta_r}, "
            f"fit_keys={list(self.fit_results.keys()) if self.fit_results else None}"
            f"{endpoint_repr})"
        )
