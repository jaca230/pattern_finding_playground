from tracklet import Tracklet
from hit import Hit
from vertex import Vertex
from pattern import Pattern
from typing import Set, Callable, List
import ROOT  # Assuming you're using the ROOT Python interface

class PatternFindingHelpers:
    def __init__(self, form_vertices_fn: Callable[[Set[Tracklet]], Set[Vertex]], form_patterns_fn: Callable[[Set[Vertex]], Set[Pattern]]):
        # Directly assign the function to public variables
        self.form_vertices_fn = form_vertices_fn
        self.form_patterns_fn = form_patterns_fn

    def form_vertices(self, tracklets: Set[Tracklet]) -> Set[Vertex]:
        """Use the stored function to form vertices from tracklets."""
        return self.form_vertices_fn(tracklets)

    def form_patterns(self, vertices: Set[Vertex]) -> Set[Pattern]:
        """Use the stored function to form patterns from vertices."""
        return self.form_patterns_fn(vertices)

    def create_tracklets(self, file, entry_index: int) -> List[Tracklet]:
        """
        Creates Tracklet objects from event data in the file.
    
        Args:
            file: The ROOT file containing event data.
            entry_index: The index of the entry to retrieve from the tree.
    
        Returns:
            A list of Tracklet objects.
        """
        # Get the tree and geometry helper from the ROOT file
        tree = file.Get("rec")
        geoHelper = file.Get("PIMCGeoHelper")
        tree.GetEntry(entry_index)
    
        tracklets = []
        tracklet_counter = 0  # Global counter for unique tracklet IDs
    
        # Loop through patterns in the event
        for pattern in tree.patternVec:
            indices = pattern.GetTrackletIndices()
            for index in indices:
                tracklet = tree.trackletVec[index]
                tracklet_id = tracklet_counter  # Use the global counter for unique ID
                tracklet_counter += 1  # Increment for the next tracklet
                particle_id = tracklet.GetPID()
                e_id = tracklet.GetEID()
                hits = []
    
                # Create Hit objects for each hit in the tracklet
                for hit in tracklet.GetAllHits():
                    
                    # Get volume names
                    vid = hit.GetVID()
                    vname = geoHelper.GetVolumeName(vid).Data()

                    if 'atar' not in vname:
                        continue
    
                    # Determine front or back side
                    detector_side = None
                    if len(vname) > 11:
                        if vname[11] == 'f':
                            detector_side = 'front'
                        elif vname[11] == 'b':
                            detector_side = 'back'

                    hit_pid = hit.GetPID()
                    hit_time = hit.GetObservedTime()
                    hit_energy = hit.GetObservedEdep()
    
                    x = geoHelper.GetX(vid)
                    y = geoHelper.GetY(vid)
                    z = geoHelper.GetZ(vid) + 0.07  # Optional Z shift
    
                    # Create Hit object
                    hit_obj = Hit(
                        z=z,
                        x=x,
                        y=y,
                        time=hit_time,
                        energy=hit_energy,
                        particle_id=hit_pid,
                        detector_side=detector_side
                    )
                    hits.append(hit_obj)
    
                # Create and store the Tracklet
                tracklet_obj = Tracklet(
                    tracklet_id=tracklet_id,
                    particle_id=particle_id,
                    e_id=e_id,
                    hits=hits
                )
                tracklets.append(tracklet_obj)
    
        return tracklets



