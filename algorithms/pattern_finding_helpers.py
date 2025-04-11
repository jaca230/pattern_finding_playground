from typing import Set, List
from models.hit import Hit
from models.tracklet import Tracklet
from models.vertex import Vertex
from models.pattern import Pattern
from algorithms.vertex.vertex_former import VertexFormer
from algorithms.pattern.pattern_former import PatternFormer

class PatternFindingHelpers:
    def __init__(self, vertex_former: VertexFormer, pattern_former: PatternFormer):
        self.vertex_former = vertex_former
        self.pattern_former = pattern_former

    def form_vertices(self, tracklets: Set[Tracklet]) -> Set[Vertex]:
        return self.vertex_former.form_vertices(tracklets)

    def form_patterns(self, vertices: Set[Vertex]) -> Set[Pattern]:
        return self.pattern_former.form_patterns(vertices)

    def create_tracklets(self, file, entry_index: int) -> List[Tracklet]:
        tree = file.Get("rec")
        geoHelper = file.Get("PIMCGeoHelper")
        tree.GetEntry(entry_index)

        tracklets = []
        tracklet_counter = 0

        for pattern in tree.patternVec:
            indices = pattern.GetTrackletIndices()
            for index in indices:
                tracklet = tree.trackletVec[index]
                tracklet_id = tracklet_counter
                tracklet_counter += 1
                particle_id = tracklet.GetPID()
                e_id = tracklet.GetEID()
                hits = []

                for hit in tracklet.GetAllHits():
                    vid = hit.GetVID()
                    vname = geoHelper.GetVolumeName(vid).Data()

                    if 'atar' not in vname:
                        continue

                    detector_side = None
                    if len(vname) > 11:
                        if vname[11] == 'f':
                            detector_side = 'front'
                        elif vname[11] == 'b':
                            detector_side = 'back'

                    hit_obj = Hit(
                        z=geoHelper.GetZ(vid) + 0.07,
                        x=geoHelper.GetX(vid),
                        y=geoHelper.GetY(vid),
                        time=hit.GetObservedTime(),
                        energy=hit.GetObservedEdep(),
                        particle_id=hit.GetPID(),
                        detector_side=detector_side
                    )
                    hits.append(hit_obj)

                tracklets.append(Tracklet(
                    tracklet_id=tracklet_id,
                    particle_id=particle_id,
                    e_id=e_id,
                    hits=hits
                ))

        return tracklets
