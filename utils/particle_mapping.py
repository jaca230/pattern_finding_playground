# Particle ID to name mapping
particle_name_map = {
    -11: {'name': 'e^{+}', 'color': '#D62728'},     # Electron (Red)
    11: {'name': 'e^{-}', 'color': '#1F77B4'},      # Positron (Blue)
    211: {'name': '\pi^{+}', 'color': '#2CA02C'},   # Pi+ (Green)
    -211: {'name': '\pi^{-}', 'color': '#9467BD'},  # Pi- (Purple)
    13: {'name': '\mu^{-}', 'color': '#FF7F0E'},    # Mu+ (Orange)
    -13: {'name': '\mu^{+}', 'color': '#8C564B'},   # Mu- (Brown)
    22: {'name': 'γ', 'color': '#17BECF'},          # Photon (Cyan)
    1000140280: {'name': '^{28}\\text{Si}', 'color': '#E377C2'},  # Silicon-28 (Magenta)
    'default': {'name': 'Unknown', 'color': '#B0B0B0'},  # Default color for unknown particles (Gray)
}
