import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors  # Correct import for color handling

def darken_color(color: str, factor: float = 0.6) -> str:
    """Darken the color by the given factor (default is 50%)."""
    color = np.array(mcolors.hex2color(color))  # Convert to RGB using matplotlib.colors
    darkened_color = np.clip(color * factor, 0, 1)  # Scale and clip to valid range
    return mcolors.rgb2hex(darkened_color)  # Convert back to hex using matplotlib.colors

def plot_tracklets(tracklets, vertices=None):
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

    # Set grid lines and vertical positions for the detectors
    dz_total = 0.139 + 0.149
    vertical_positions = np.arange(0, 7, dz_total / 2.0)
    for axis in ax:
        for z in vertical_positions:
            axis.axvline(x=z, color='gray', linestyle='--', alpha=0.5)

    # Store plotted particle types for legend
    plotted_particles = set()

    # Plot each tracklet
    for tracklet in tracklets:
        particle_color = tracklet.particle_color

        ax[0].scatter(
            [hit.z for hit in tracklet.get_front_hits()],
            [hit.x for hit in tracklet.get_front_hits()],
            color=particle_color, alpha=0.7, label=f"${tracklet.particle_name}$"
        )

        ax[1].scatter(
            [hit.z for hit in tracklet.get_back_hits()],
            [hit.y for hit in tracklet.get_back_hits()],
            color=particle_color, alpha=0.7, label=f"${tracklet.particle_name}$"
        )

        ax[2].scatter(
            [hit.z for hit in tracklet.hits],
            [hit.energy for hit in tracklet.hits],
            color=particle_color, alpha=0.7, label=f"${tracklet.particle_name}$"
        )

        endpoint_0, endpoint_1 = tracklet.get_endpoints()
        if endpoint_0 and endpoint_1:
            ax[0].scatter(endpoint_0.z, endpoint_0.x, color=particle_color, s=150, marker='*', zorder=5, alpha=0.4)
            ax[0].scatter(endpoint_1.z, endpoint_1.x, color=particle_color, s=150, marker='*', zorder=5, alpha=0.4)
            ax[1].scatter(endpoint_0.z, endpoint_0.y, color=particle_color, s=150, marker='*', zorder=5, alpha=0.4)
            ax[1].scatter(endpoint_1.z, endpoint_1.y, color=particle_color, s=150, marker='*', zorder=5, alpha=0.4)

            dark_particle_color = darken_color(particle_color, factor=0.3)
            ax[0].plot([endpoint_0.z, endpoint_1.z], [endpoint_0.x, endpoint_1.x], color=dark_particle_color, linestyle='-', linewidth=3, alpha=0.7)
            ax[1].plot([endpoint_0.z, endpoint_1.z], [endpoint_0.y, endpoint_1.y], color=dark_particle_color, linestyle='-', linewidth=3, alpha=0.7)

        plotted_particles.add(tracklet.particle_name)

    # Plot centroids from vertices, if provided
    if vertices is not None:
        for vertex in vertices:
            # Plot front centroid on x-z plane (subplot 0)
            if 'front' in vertex.tracklet_former_results and 'centroid' in vertex.tracklet_former_results['front']:
                front_centroid = vertex.tracklet_former_results['front']['centroid']
                ax[0].scatter(
                    front_centroid[2], front_centroid[0],
                    color='magenta', edgecolors='black', marker='X',
                    s=150, linewidths=1.5, zorder=15, label='Front Centroid'
                )

            # Plot back centroid on y-z plane (subplot 1)
            if 'back' in vertex.tracklet_former_results and 'centroid' in vertex.tracklet_former_results['back']:
                back_centroid = vertex.tracklet_former_results['back']['centroid']
                ax[1].scatter(
                    back_centroid[2], back_centroid[1],
                    color='magenta', edgecolors='black', marker='X',
                    s=150, linewidths=1.5, zorder=15, label='Back Centroid'
                )

    # Set log scale for energy plot
    ax[2].set_yscale('log')

    # Add a legend to the energy plot
    ax[2].legend(title="Tracklet Types", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

    # Adjust layout and show plot
    plt.subplots_adjust(hspace=0.4, right=0.75)
    plt.tight_layout()
    plt.show()
    return fig, ax

