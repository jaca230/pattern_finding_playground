import os
from datetime import datetime
import numpy as np
from typing import List, Tuple, Set, Optional
from models.point_3d import Point3D
from models.tracklet import Tracklet
from models.vertex import Vertex
from models.hit import Hit
from models.event_patterns import EventPatterns

def print_file_creation_time(file_name):
    """Print the creation time of a file."""
    if os.path.exists(file_name):
        creation_time = os.path.getctime(file_name)
        formatted_time = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"The file '{file_name}' was created on: {formatted_time}")
    else:
        print(f"File '{file_name}' does not exist.")


def linear_fit(x: List[float], y: List[float]) -> Tuple[float, float, float, float]:
    """
    Perform a linear fit y = mx + b, and return the slope (m), intercept (b)
    and their uncertainties (unc_m, unc_b).
    
    Arguments:
    x -- List of x values
    y -- List of y values
    
    Returns:
    m, b, unc_m, unc_b -- Slope, intercept, and their uncertainties
    """
    # Convert lists to NumPy arrays for better compatibility with NumPy functions
    x = np.array(x)
    y = np.array(y)
    
    # Handle the case where there are exactly two points
    if len(x) == 2:
        # If there are exactly two points, we can directly calculate the slope and intercept
        m = (y[1] - y[0]) / (x[1] - x[0])
        b = y[0] - m * x[0]
        # For two points, there's no uncertainty
        unc_m = 0
        unc_b = 0
    else:
        # Perform linear fit for more than two points
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Calculate residuals and sum of squares
        residuals = y - (m * x + b)
        
        # Calculate uncertainties for slope and intercept
        unc_m = np.sqrt(np.sum(residuals**2) / (len(x) - 2)) / np.sqrt(np.sum((x - np.mean(x))**2))
        unc_b = unc_m * np.sqrt(np.sum(x**2) / len(x))
    
    return m, b, unc_m, unc_b

def fit_tracklet_hits(hits: List[Hit]) -> dict:
    """
    Perform linear fits for hits in both detector sides and return the results.
    
    Args:
        hits: List of Hit objects.
        
    Returns:
        Dictionary containing fit results for x(z) and y(z).
    """

    # Separate hits by detector side
    front_hits = [hit for hit in hits if hit.detector_side == 'front']
    back_hits = [hit for hit in hits if hit.detector_side == 'back']

    # Get the min and max z values across all hits
    min_z = min(hit.z for hit in hits)
    max_z = max(hit.z for hit in hits)

    fit_results = {
        "min_z": min_z,
        "max_z": max_z
    }

    # Back hits (Y vs Z)
    back_zs = [hit.z for hit in back_hits]
    back_ys = [hit.y for hit in back_hits]
    if len(set(back_zs)) > 1: #at least 2 unique z values
        m, b, unc_m, unc_b = linear_fit(back_zs, back_ys)
        y_min = m * min_z + b
        y_max = m * max_z + b
    elif len(back_zs) > 0: # not 2 unique z values, but at least 1 hit
        m = b = unc_m = unc_b = None
        y_val = np.mean(back_ys)
        y_min = y_max = y_val
    else: # no hits
        m = b = unc_m = unc_b = y_min = y_max = None
    fit_results['y_z_fit'] = {
        "m": m, "b": b, "unc_m": unc_m, "unc_b": unc_b,
        "y_min": y_min, "y_max": y_max
    }

    # Front hits (X vs Z)
    front_zs = [hit.z for hit in front_hits]
    front_xs = [hit.x for hit in front_hits]
    if len(set(front_zs)) > 1: #at least 2 unique z values
        m, b, unc_m, unc_b = linear_fit(front_zs, front_xs)
        x_min = m * min_z + b
        x_max = m * max_z + b
    elif len(front_zs) > 0: # not 2 unique z values, but at least 1 hit
        m = b = unc_m = unc_b = None
        x_val = np.mean(front_xs)
        x_min = x_max = x_val
    else: # no hits
        m = b = unc_m = unc_b = x_min = x_max = None
    fit_results['x_z_fit'] = {
        "m": m, "b": b, "unc_m": unc_m, "unc_b": unc_b,
        "x_min": x_min, "x_max": x_max
    }

    return fit_results

