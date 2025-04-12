from typing import List, Any
from models.tracklet import Tracklet
from models.hit import Hit
from algorithms.tracklet.tracklet_former import TrackletFormer  # your abstract base

class DefaultTrackletFormer(TrackletFormer):
    def form_tracklets(self, file: Any, entry_index: int) -> List[Tracklet]:
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

        result_info = {}

        return tracklets, result_info
