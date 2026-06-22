PARTICLE_STYLES = {
    -11: {"name": r"e^{+}", "color": "#D62728"},
    11: {"name": r"e^{-}", "color": "#1F77B4"},
    211: {"name": r"\pi^{+}", "color": "#2CA02C"},
    -211: {"name": r"\pi^{-}", "color": "#9467BD"},
    13: {"name": r"\mu^{-}", "color": "#FF7F0E"},
    -13: {"name": r"\mu^{+}", "color": "#8C564B"},
    22: {"name": r"\gamma", "color": "#17BECF"},
    1000140280: {"name": r"^{28}\mathrm{Si}", "color": "#E377C2"},
    "default": {"name": "Unknown", "color": "#B0B0B0"},
}

# Backward-compatible name for code that wants the raw mapping.
particle_name_map = PARTICLE_STYLES


def particle_style(particle_id: int) -> dict:
    return PARTICLE_STYLES.get(particle_id, PARTICLE_STYLES["default"])
