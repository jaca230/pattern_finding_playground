from .data_cut import DataCut
from .registry import register_cut


@register_cut(
    name="allowed_atar_particles",
    description="Reject entries containing reconstructed ATAR hits from particles outside a configurable allowed PID set.",
    parameters={
        "allowed_particle_ids": "Iterable of integer particle IDs allowed to appear in reconstructed ATAR hits.",
        "enabled": "Set False to leave the cut configured but inactive.",
    },
    example="AllowedAtarParticlesCut(allowed_particle_ids={211, -211, 13, -13, 11, -11}, enabled=True)",
)
class AllowedAtarParticlesCut(DataCut):
    """Require all reconstructed ATAR hit particle IDs to come from an allowed set."""

    def __init__(
        self,
        *,
        allowed_particle_ids: set[int] | tuple[int, ...] | list[int] = (211, -211, 13, -13, 11, -11),
        enabled: bool = True,
    ):
        super().__init__(name="allowed_atar_particles", enabled=enabled)
        self.allowed_particle_ids = {int(pid) for pid in allowed_particle_ids}

    def accepts(self, data_file, entry: dict) -> bool:
        hits = entry.get("hits")
        if hits is None:
            return True

        for hit in hits:
            if int(hit.GetPID()) not in self.allowed_particle_ids:
                return False
        return True
