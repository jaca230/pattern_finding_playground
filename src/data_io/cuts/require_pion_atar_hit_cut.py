from .data_cut import DataCut
from .registry import register_cut


@register_cut(
    name="require_pion_atar_hit",
    description="Reject entries that do not contain at least one reconstructed ATAR hit attributed to a pion.",
    parameters={
        "pion_particle_ids": "Iterable of integer particle IDs to count as pion-like reconstructed ATAR hits.",
        "enabled": "Set False to leave the cut configured but inactive.",
    },
    example="RequirePionAtarHitCut(pion_particle_ids={211, -211}, enabled=True)",
)
class RequirePionAtarHitCut(DataCut):
    """Require at least one reconstructed ATAR hit with a pion particle ID."""

    def __init__(
        self,
        *,
        pion_particle_ids: set[int] | tuple[int, ...] | list[int] = (211, -211),
        enabled: bool = True,
    ):
        super().__init__(name="require_pion_atar_hit", enabled=enabled)
        self.pion_particle_ids = {int(pid) for pid in pion_particle_ids}

    def accepts(self, data_file, entry: dict) -> bool:
        hits = entry.get("hits")
        if hits is None:
            return False

        for hit in hits:
            if int(hit.GetPID()) in self.pion_particle_ids:
                return True
        return False
