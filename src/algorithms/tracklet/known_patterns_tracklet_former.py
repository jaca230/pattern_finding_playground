from typing import List, Any, Optional
from collections import Counter
from models.tracklet import Tracklet
from models.hit import Hit
from algorithms.tracklet.tracklet_former import TrackletFormer


class KnownPatternsTrackletFormer(TrackletFormer):
    def form_tracklets(
        self,
        reco_event: Any,
        geoHelper: Any,
        storage: Optional[Any] = None,
        truth_event: Optional[Any] = None,
    ) -> tuple[List[Tracklet], dict]:
        tracklets = []
        tracklet_idx_to_id = {}
        patterns_reco = {}

        tracklet_counter = 0

        # First loop: Over the reconstructed (reco) patterns
        for pattern_idx, pattern in enumerate(reco_event.patterns):
            indices = pattern.GetTrackletIndices()
            pattern_tracklet_ids = []

            # For each tracklet in the reco pattern
            for idx in indices:
                idx = int(idx)
                raw_tracklet = reco_event.tracklets[idx]
                particle_id = raw_tracklet.pid
                e_id = raw_tracklet.eid
                hits = []

                # Build the hits for the tracklet
                for hit in raw_tracklet.hits:
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

                # Create the tracklet object
                tracklet = Tracklet(
                    tracklet_id=tracklet_counter,
                    particle_id=particle_id,
                    e_id=e_id,
                    hits=hits
                )

                # Assign the reco pattern index to the tracklet's extra_info
                tracklet.extra_info["pattern_reco"] = pattern_idx
                tracklets.append(tracklet)

                # Track the relationship between pattern and tracklet
                tracklet_idx_to_id[idx] = tracklet_counter
                pattern_tracklet_ids.append(tracklet_counter)

                tracklet_counter += 1

            patterns_reco[pattern_idx] = pattern_tracklet_ids

        # Second loop: Over the truth patterns
        truth_source = truth_event if truth_event is not None else reco_event

        n_patterns_truth = 0
        particles_counter = Counter()
        patterns_truth = {}

        for pattern_idx, pattern in enumerate(truth_source.patterns):
            n_patterns_truth += 1
            indices = pattern.GetTrackletIndices()
            pattern_tracklet_ids = []

            for index in indices:
                index = int(index)
                tracklet = truth_source.tracklets[index]
                particle_id = tracklet.pid

                particles_counter[particle_id] += 1
                pattern_tracklet_ids.append(index)

            patterns_truth[pattern_idx] = pattern_tracklet_ids

        # Prepare the result information with the truth patterns and reconstructed patterns
        result_info = {
            "n_patterns_truth": n_patterns_truth,
            "particles_in_event_truth": particles_counter,
            "patterns_truth": patterns_truth,
            "patterns_reco": patterns_reco
        }

        return tracklets, result_info
