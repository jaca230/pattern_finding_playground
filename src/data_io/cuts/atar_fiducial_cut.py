from .registry import register_cut
from .data_cut import DataCut


@register_cut(
    name="atar_fiducial",
    description="Require reconstructed ATAR tracklet hits to stay inside a simple front/back fiducial window.",
    parameters={
        "max_abs_coordinate": "Maximum allowed |x| or |y| in mm for front/back ATAR hits.",
        "enabled": "Set False to keep the cut in the list without applying it.",
    },
    example="AtarFiducialCut(max_abs_coordinate=8.0, enabled=True)",
)
class AtarFiducialCut(DataCut):
    """Require ATAR tracklet hits to stay inside a simple front/back fiducial window."""

    def __init__(self, *, max_abs_coordinate: float = 8.0, enabled: bool = True):
        super().__init__(name="atar_fiducial", enabled=enabled)
        self.max_abs_coordinate = max_abs_coordinate

    def accepts(self, data_file, entry: dict) -> bool:
        geo = data_file.geo
        hits = entry["hits"]

        for tracklet in entry["tracklets"]:
            for hit_index in tracklet.GetAtarHitIndices():
                hit_index = int(hit_index)
                if hit_index < 0 or hit_index >= hits.size():
                    continue

                hit = hits[hit_index]
                volume_id = hit.GetVID()
                volume_name = geo.GetVolumeName(volume_id).Data()
                if "atar" not in volume_name or len(volume_name) <= 11:
                    continue

                side = volume_name[11]
                if side == "f" and abs(geo.GetX(volume_id)) > self.max_abs_coordinate:
                    return False
                if side == "b" and abs(geo.GetY(volume_id)) > self.max_abs_coordinate:
                    return False

        return True
