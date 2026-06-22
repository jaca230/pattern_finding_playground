from typing import List, Any, Optional
from collections import Counter
from models.tracklet import Tracklet
from models.hit import Hit
from algorithms.tracklet.tracklet_former import TrackletFormer

class DefaultTrackletFormer(TrackletFormer):
    def form_tracklets(
        self,
        reco_entry: Any,
        geo: Any,
        storage: Optional[Any] = None,
        truth_entry: Optional[Any] = None,
    ) -> tuple[List[Tracklet], dict]:
        tracklets = []
        tracklet_counter = 0

        for index, tracklet in enumerate(reco_entry["tracklets"]):
            particle_id = int(tracklet.GetPID())
            e_id = int(tracklet.GetEID())
            hits = []

            for hit in self._root_tracklet_hits(tracklet, reco_entry["hits"]):
                vid = hit.GetVID()
                vname = geo.GetVolumeName(vid).Data()

                if 'atar' not in vname:
                    continue

                detector_side = None
                if len(vname) > 11:
                    if vname[11] == 'f':
                        detector_side = 'front'
                    elif vname[11] == 'b':
                        detector_side = 'back'

                hit_obj = Hit(
                    z=geo.GetZ(vid) + 0.07,
                    x=geo.GetX(vid),
                    y=geo.GetY(vid),
                    time=hit.GetObservedTime(),
                    energy=hit.GetObservedEdep(),
                    particle_id=hit.GetPID(),
                    detector_side=detector_side
                )
                hits.append(hit_obj)

            tracklet_id = tracklet_counter
            tracklet_counter += 1

            tracklets.append(Tracklet(
                tracklet_id=tracklet_id,
                particle_id=particle_id,
                e_id=e_id,
                hits=hits
            ))

        truth_source = truth_entry if truth_entry is not None else reco_entry

        n_patterns_truth = 0
        particles_counter = Counter()
        patterns_truth = {}

        for pattern_idx, pattern in enumerate(truth_source["patterns"]):
            n_patterns_truth += 1
            indices = pattern.GetTrackletIndices()
            pattern_tracklet_ids = []

            for index in indices:
                tracklet = truth_source["tracklets"][int(index)]
                particle_id = int(tracklet.GetPID())

                particles_counter[particle_id] += 1
                pattern_tracklet_ids.append(int(index))

            patterns_truth[pattern_idx] = pattern_tracklet_ids

        result_info = {
            "n_patterns_truth": n_patterns_truth,
            "particles_in_event_truth": particles_counter,
            "patterns_truth": patterns_truth
        }

        return tracklets, result_info
