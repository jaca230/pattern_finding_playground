import numpy as np
from typing import List, Tuple

from models.hit import Hit


def linear_fit(x: List[float], y: List[float]) -> Tuple[float, float, float, float]:
    """Perform a linear fit y = mx + b and return slope/intercept plus uncertainties."""
    x = np.array(x)
    y = np.array(y)

    if len(x) == 2:
        m = (y[1] - y[0]) / (x[1] - x[0])
        b = y[0] - m * x[0]
        unc_m = 0
        unc_b = 0
    else:
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        residuals = y - (m * x + b)
        unc_m = np.sqrt(np.sum(residuals**2) / (len(x) - 2)) / np.sqrt(np.sum((x - np.mean(x))**2))
        unc_b = unc_m * np.sqrt(np.sum(x**2) / len(x))

    return m, b, unc_m, unc_b


def fit_tracklet_hits(hits: List[Hit]) -> dict:
    """Fit front and back ATAR hit projections as linear functions of z."""
    front_hits = [hit for hit in hits if hit.detector_side == "front"]
    back_hits = [hit for hit in hits if hit.detector_side == "back"]

    min_z = min(hit.z for hit in hits)
    max_z = max(hit.z for hit in hits)
    fit_results = {"min_z": min_z, "max_z": max_z}

    back_zs = [hit.z for hit in back_hits]
    back_ys = [hit.y for hit in back_hits]
    if len(set(back_zs)) > 1:
        m, b, unc_m, unc_b = linear_fit(back_zs, back_ys)
        y_min = m * min_z + b
        y_max = m * max_z + b
    elif len(back_zs) > 0:
        m = b = unc_m = unc_b = None
        y_val = np.mean(back_ys)
        y_min = y_max = y_val
    else:
        m = b = unc_m = unc_b = y_min = y_max = None
    fit_results["y_z_fit"] = {
        "m": m,
        "b": b,
        "unc_m": unc_m,
        "unc_b": unc_b,
        "y_min": y_min,
        "y_max": y_max,
    }

    front_zs = [hit.z for hit in front_hits]
    front_xs = [hit.x for hit in front_hits]
    if len(set(front_zs)) > 1:
        m, b, unc_m, unc_b = linear_fit(front_zs, front_xs)
        x_min = m * min_z + b
        x_max = m * max_z + b
    elif len(front_zs) > 0:
        m = b = unc_m = unc_b = None
        x_val = np.mean(front_xs)
        x_min = x_max = x_val
    else:
        m = b = unc_m = unc_b = x_min = x_max = None
    fit_results["x_z_fit"] = {
        "m": m,
        "b": b,
        "unc_m": unc_m,
        "unc_b": unc_b,
        "x_min": x_min,
        "x_max": x_max,
    }

    return fit_results
