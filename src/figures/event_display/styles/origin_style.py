from __future__ import annotations


ORIGIN_STYLES = {
    "primary": {"label": "primary", "color": "#1F77B4"},
    "pileup": {"label": "pileup", "color": "#2CA02C"},
    "radioactivity": {"label": "radioactivity", "color": "#D62728"},
    "unknown": {"label": "unknown", "color": "#777777"},
}


def origin_style(origin: str | None) -> dict:
    return ORIGIN_STYLES.get(origin or "unknown", ORIGIN_STYLES["unknown"])
