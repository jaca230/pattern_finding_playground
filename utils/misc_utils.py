import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

def print_file_creation_time(file_name):
    """Print the creation time of a file."""
    if os.path.exists(file_name):
        creation_time = os.path.getctime(file_name)
        formatted_time = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"The file '{file_name}' was created on: {formatted_time}")
    else:
        print(f"File '{file_name}' does not exist.")

def dummy_form_vertices(tracklets):
    return set()

def dummy_form_patterns(vertices):
    return set()

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
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    residuals = y - (m * np.array(x) + b)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    unc_m = np.sqrt(np.sum(residuals**2) / (len(x) - 2)) / np.sqrt(np.sum((x - np.mean(x))**2))
    unc_b = unc_m * np.sqrt(np.sum(x**2) / len(x))
    
    return m, b, unc_m, unc_b


class Point3D:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
    
    def __repr__(self):
        return f"Point3D(x={self.x}, y={self.y}, z={self.z})"


def plot_tracklets(tracklets):
    # Create a figure with 3 subplots (1 row, 3 columns)
    fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Set axis labels
    ax[0].set_ylabel("x [mm]")
    ax[1].set_ylabel("y [mm]")
    ax[2].set_xlabel("z [mm]")
    ax[2].set_ylabel("Energy [MeV]")

    # Set x-axis limits
    for axis in ax:
        axis.set_xlim([0, 7])

    # Set grid lines and vertical positions for the detectors (based on a sample dz_total of 0.139 + 0.149)
    dz_total = 0.139 + 0.149
    vertical_positions = np.arange(0, 7, dz_total / 2.0)  # Adjust based on detector positions
    for axis in ax:
        for z in vertical_positions:
            axis.axvline(x=z, color='gray', linestyle='--', alpha=0.5)

    # Store plotted particle types for legend
    plotted_particles = set()

    # Iterate through each tracklet to extract hit data
    for tracklet in tracklets:
        particle_color = tracklet.particle_color  # Get the particle color for each tracklet

        # Plot hits for front plane (x vs z) and back plane (y vs z)
        ax[0].scatter(
            [hit.z for hit in tracklet.get_front_hits()],  # z values for front hits
            [hit.x for hit in tracklet.get_front_hits()],  # x values for front hits
            color=particle_color, alpha=0.7, label=f"${tracklet.particle_name}$ (front)"
        )

        ax[1].scatter(
            [hit.z for hit in tracklet.get_back_hits()],  # z values for back hits
            [hit.y for hit in tracklet.get_back_hits()],  # y values for back hits
            color=particle_color, alpha=0.7, label=f"${tracklet.particle_name}$ (back)"
        )

        # Plot energy vs z for all hits
        ax[2].scatter(
            [hit.z for hit in tracklet.hits],  # z values for all hits
            [hit.energy for hit in tracklet.hits],  # energy values for all hits
            color=particle_color, alpha=0.7
        )

        # Track unique particle types for the legend
        plotted_particles.add(tracklet.particle_name)

    # Set log scale for energy plot
    ax[2].set_yscale('log')

    # Add a legend to the first plot to show tracklet names
    ax[0].legend(title="Tracklet Types", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

    # Adjust layout and add space for the legend
    plt.subplots_adjust(hspace=0.4, right=0.75)

    # Show the plot
    plt.tight_layout()
    plt.show()



