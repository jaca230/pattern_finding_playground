from __future__ import annotations

import numpy as np
from matplotlib.colors import LogNorm, Normalize


class EnergyStyle:
    def __init__(self, values, *, cmap="viridis", floor=1.0e-4):
        self.cmap = cmap
        finite_values = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
        finite_values = finite_values[finite_values > 0]
        if finite_values.size:
            vmin = max(float(np.nanmin(finite_values)), floor)
            vmax = max(float(np.nanmax(finite_values)), vmin * 1.01)
            self.norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            self.norm = Normalize(vmin=floor, vmax=floor * 10.0)
