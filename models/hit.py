from typing import Optional
from utils.particle_mapping import particle_name_map  # Assuming this file is named particle_mapping.py

class Hit:
    def __init__(
        self,
        z: float,
        particle_id: int,
        x: Optional[float] = None,
        y: Optional[float] = None,
        time: Optional[float] = None,
        energy: Optional[float] = None,
        detector_side: Optional[str] = None  # Can be 'f' or 'b' indicating front or back
    ):
        """
        Represents a single detector hit in a strip-based system (x-z or y-z).

        Args:
            z: Position along the beam axis (always present).
            particle_id: The ID of the particle that caused the hit.
            x: X position if it's an x-strip hit (optional).
            y: Y position if it's a y-strip hit (optional).
            time: Timestamp of the hit (optional).
            energy: Energy associated with the hit (optional).
            detector_side: The side of the detector ('front' or 'back') (optional).
        """
        if x is None and y is None:
            raise ValueError("Either x or y must be provided.")
        
        # Ensure we set the side of the detector
        if detector_side not in ['front', 'back', None]:
            raise ValueError("detector_side must be either 'f' (front) or 'b' (back).")
        
        self.particle_id = particle_id
        self.particle_name, self.particle_color = self.get_particle_info(particle_id)

        self.x = x
        self.y = y
        self.z = z
        self.time = time
        self.energy = energy
        self.detector_side = detector_side

    def get_particle_info(self, particle_id: int):
        """Retrieves the particle name and color based on the particle ID."""
        particle_info = particle_name_map.get(particle_id, particle_name_map['default'])
        return particle_info['name'], particle_info['color']

    def __repr__(self) -> str:
        return (
            f"Hit(particle_id={self.particle_id}, name={self.particle_name}, "
            f"color={self.particle_color}, x={self.x}, y={self.y}, z={self.z}, "
            f"time={self.time}, energy={self.energy}, detector_side={self.detector_side})"
        )
