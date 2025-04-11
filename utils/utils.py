import os
from datetime import datetime
import numpy as np
from typing import List, Tuple, Set, Optional
from models.point_3d import Point3D
from models.tracklet import Tracklet
from models.vertex import Vertex
from models.hit import Hit

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
    This function takes a list of hits, performs the fit for both front and back hits, 
    and returns the fit results as a dictionary.
    
    Args:
        hits: List of Hit objects.
        
    Returns:
        Dictionary containing the fit results.
    """
    # Separate hits by detector side
    front_hits = [hit for hit in hits if hit.detector_side == 'front']
    back_hits = [hit for hit in hits if hit.detector_side == 'back']

    # Get the min and max z values (used for both front and back hits)
    min_z = min(hit.z for hit in hits)
    max_z = max(hit.z for hit in hits)

    # Initialize fit results dictionary
    fit_results = {
        "min_z": min_z,
        "max_z": max_z
    }

    # Handle back hits
    if len(back_hits) >= 2:
        X_to_fit = [hit.z for hit in back_hits]
        Y_to_fit = [hit.y for hit in back_hits]
        m, b, unc_m, unc_b = linear_fit(X_to_fit, Y_to_fit)
        y_min = m * min_z + b
        y_max = m * max_z + b
        fit_results['y_z_fit'] = {
            "m": m, 
            "b": b, 
            "unc_m": unc_m, 
            "unc_b": unc_b,
            "y_min": y_min,
            "y_max": y_max
        }

    elif len(back_hits) == 1:
        y_min = back_hits[0].y
        y_max = back_hits[0].y
        fit_results['y_z_fit'] = {
            "m": None, 
            "b": None, 
            "unc_m": None, 
            "unc_b": None,
            "y_min": y_min,
            "y_max": y_max
        }

    else:
        # No back hits, use default values
        fit_results['y_z_fit'] = {
            "m": None, 
            "b": None, 
            "unc_m": None, 
            "unc_b": None,
            "y_min": None,
            "y_max": None
        }

    # Handle front hits
    if len(front_hits) >= 2:
        X_to_fit = [hit.z for hit in front_hits]
        Y_to_fit = [hit.x for hit in front_hits]
        m, b, unc_m, unc_b = linear_fit(X_to_fit, Y_to_fit)
        x_min = m * min_z + b
        x_max = m * max_z + b
        fit_results['x_z_fit'] = {
            "m": m, 
            "b": b, 
            "unc_m": unc_m, 
            "unc_b": unc_b,
            "x_min": x_min,
            "x_max": x_max
        }

    elif len(front_hits) == 1:
        x_min = front_hits[0].x
        x_max = front_hits[0].x
        fit_results['x_z_fit'] = {
            "m": None, 
            "b": None, 
            "unc_m": None, 
            "unc_b": None,
            "x_min": x_min,
            "x_max": x_max
        }

    else:
        # No front hits, use default values
        fit_results['x_z_fit'] = {
            "m": None, 
            "b": None, 
            "unc_m": None, 
            "unc_b": None,
            "x_min": None,
            "x_max": None
        }

    return fit_results