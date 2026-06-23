from .data_cut import DataCut
from .registry import register_cut


@register_cut(
    name="nonempty_event",
    description="Reject entries with no reconstructed ATAR hits.",
    parameters={
        "require_atar_hits": "If True, require at least one reconstructed ATAR hit.",
        "enabled": "Set False to leave the cut configured but inactive.",
    },
    example="EmptyEventCut(require_atar_hits=True, enabled=True)",
)
class EmptyEventCut(DataCut):
    """Require at least some reconstructed ATAR activity in the event."""

    def __init__(self, *, require_atar_hits: bool = True, enabled: bool = True):
        super().__init__(name="nonempty_event", enabled=enabled)
        self.require_atar_hits = bool(require_atar_hits)

    def accepts(self, data_file, entry: dict) -> bool:
        if not self.require_atar_hits:
            return True

        atar_hits = entry.get("hits")
        return atar_hits is not None and int(atar_hits.size()) > 0
