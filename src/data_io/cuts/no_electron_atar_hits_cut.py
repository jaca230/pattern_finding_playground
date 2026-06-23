from .data_cut import DataCut
from .registry import register_cut


@register_cut(
    name="no_electron_atar_hits",
    description="Reject entries containing reconstructed ATAR hits attributed to electrons while allowing positrons.",
    parameters={
        "electron_particle_ids": "Iterable of integer particle IDs to treat as forbidden electron-like ATAR hits.",
        "enabled": "Set False to leave the cut configured but inactive.",
    },
    example="NoElectronAtarHitsCut(electron_particle_ids={11}, enabled=True)",
)
class NoElectronAtarHitsCut(DataCut):
    """Reject reconstructed ATAR entries that contain forbidden electron-like hit particle IDs."""

    def __init__(
        self,
        *,
        electron_particle_ids: set[int] | tuple[int, ...] | list[int] = (11,),
        enabled: bool = True,
    ):
        super().__init__(name="no_electron_atar_hits", enabled=enabled)
        self.electron_particle_ids = {int(pid) for pid in electron_particle_ids}

    def accepts(self, data_file, entry: dict) -> bool:
        hits = entry.get("hits")
        if hits is None:
            return True

        for hit in hits:
            if int(hit.GetPID()) in self.electron_particle_ids:
                return False
        return True
