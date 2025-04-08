import os
from datetime import datetime
import numpy as np
from typing import List, Tuple, Set, Optional
from point_3d import Point3D
from tracklet import Tracklet
from vertex import Vertex
from hit import Hit

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
            "y_min": 0,
            "y_max": 0
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
            "x_min": 0,
            "x_max": 0
        }

    return fit_results



def determine_endpoints(tracklet: Tracklet) -> tuple[Optional[Point3D], Optional[Point3D]]:
    """Determines the endpoints of a tracklet based on the fit results and stores the endpoints."""
    
    # Apply the fitting function
    tracklet.fitter = fit_tracklet_hits  # Assign the fitting function
    tracklet.fit()  # Fit the tracklet using the fitter
    
    # Get the fitted y_min, y_max for back hits and x_min, x_max for front hits from fit_results
    y_min = tracklet.fit_results.get("y_z_fit", {}).get("y_min", 0)
    y_max = tracklet.fit_results.get("y_z_fit", {}).get("y_max", 0)
    
    x_min = tracklet.fit_results.get("x_z_fit", {}).get("x_min", 0)
    x_max = tracklet.fit_results.get("x_z_fit", {}).get("x_max", 0)
    
    # Get the min and max z values from fit_results
    min_z = tracklet.fit_results.get("min_z", 0)
    max_z = tracklet.fit_results.get("max_z", 0)

    # Create the endpoints using the fitted values and the z values from the fit results
    endpoint_0 = Point3D(x_min, y_min, min_z)
    endpoint_1 = Point3D(x_max, y_max, max_z)

    # Set the endpoints in the tracklet
    tracklet.set_endpoints(endpoint_0, endpoint_1)

    return endpoint_0, endpoint_1








def form_vertices(tracklets: Set[Tracklet]) -> Set[Vertex]:
    vertices = set()
    
    # First pass: Determine endpoints for each tracklet
    for tracklet in tracklets:
        endpoint_0, endpoint_1 = determine_endpoints(tracklet)
        tracklet.set_endpoints(endpoint_0, endpoint_1)

    # Second pass: Create vertices based on endpoint distances
    for index, tracklet in enumerate(tracklets):
        # Create a vertex using the index as the vertex_id
        vertex = Vertex(seed_tracklet=tracklet, vertex_id=index)

        
    return vertices
