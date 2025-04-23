from typing import List, Any, Tuple, Optional
from collections import Counter
from models.tracklet import Tracklet
from models.hit import Hit
from algorithms.tracklet.tracklet_former import TrackletFormer

class DefaultTrackletFormer(TrackletFormer):
    def __init__(self, truth_tree: Optional[Any] = None):
        self.truth_tree = truth_tree

    def form_tracklets(self, tree: Any, geoHelper: Any, entry_index: int) -> Tuple[List[Tracklet], dict]:
        tree.GetEntry(entry_index)
        if self.truth_tree:
            self.truth_tree.GetEntry(entry_index)

        tracklets = []
        tracklet_counter = 0

        for index, tracklet in enumerate(tree.trackletVec):
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

            tracklet_id = tracklet_counter
            tracklet_counter += 1

            tracklets.append(Tracklet(
                tracklet_id=tracklet_id,
                particle_id=particle_id,
                e_id=e_id,
                hits=hits
            ))

        # Use truth tree if provided, otherwise default to the current tree
        truth_source = self.truth_tree if self.truth_tree is not None else tree

        n_patterns_truth = 0
        particles_counter = Counter()
        patterns_truth = {}

        for pattern_idx, pattern in enumerate(truth_source.patternVec):
            n_patterns_truth += 1
            indices = pattern.GetTrackletIndices()
            pattern_tracklet_ids = []

            for index in indices:
                tracklet = truth_source.trackletVec[index]
                particle_id = tracklet.GetPID()

                particles_counter[particle_id] += 1
                pattern_tracklet_ids.append(index)

            patterns_truth[pattern_idx] = pattern_tracklet_ids

        result_info = {
            "n_patterns_truth": n_patterns_truth,
            "particles_in_event_truth": particles_counter,
            "patterns_truth": patterns_truth
        }

        return tracklets, result_info
