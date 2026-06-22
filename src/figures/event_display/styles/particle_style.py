from __future__ import annotations

from matplotlib.colors import to_hex, to_rgb


PARTICLE_STYLES = {
    -11: {"name": r"e^{+}", "color": "#D55E00"},
    11: {"name": r"e^{-}", "color": "#A65628"},
    211: {"name": r"\pi^{+}", "color": "#E69F00"},
    -211: {"name": r"\pi^{-}", "color": "#984EA3"},
    13: {"name": r"\mu^{-}", "color": "#000000"},
    -13: {"name": r"\mu^{+}", "color": "#CC79A7"},
    22: {"name": r"\gamma", "color": "#666666"},
    1000140280: {"name": r"^{28}\mathrm{Si}", "color": "#F781BF"},
    "default": {"name": "Unknown", "color": "#666666"},
}

particle_name_map = PARTICLE_STYLES


def particle_style(particle_id: int) -> dict:
    return PARTICLE_STYLES.get(int(particle_id), PARTICLE_STYLES["default"])


def tracklet_line_color(particle_id: int) -> str:
    """Return a slightly darkened variant so the fitted path reads above hit fills."""
    base = to_rgb(particle_style(particle_id)["color"])
    shifted = tuple(max(0.0, 0.68 * channel) for channel in base)
    return to_hex(shifted)
