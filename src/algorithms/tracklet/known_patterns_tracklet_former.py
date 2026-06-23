from typing import List, Any, Optional
from collections import Counter
from algorithms.registry import register_algorithm
from models.tracklet import Tracklet
from models.hit import Hit
from algorithms.tracklet.tracklet_former import TrackletFormer


@register_algorithm(
    "tracklet",
    name="known_patterns",
    description="Build tracklets by following existing reconstructed pattern membership from the input file.",
)
class KnownPatternsTrackletFormer(TrackletFormer):
    def form_tracklets(
        self,
        event_entry: Any,
        geo: Any,
        storage: Optional[Any] = None,
        reference_truth_entry: Optional[Any] = None,
    ) -> tuple[List[Tracklet], dict]:
        tracklets = []
        patterns_reco = {}

        tracklet_counter = 0

        # First loop: build analysis tracklets from reconstructed pattern membership.
        for pattern_idx, pattern in enumerate(event_entry["patterns"]):
            indices = pattern.GetTrackletIndices()
            pattern_tracklet_ids = []

            for idx in indices:
                idx = int(idx)
                raw_tracklet = event_entry["tracklets"][idx]
                particle_id = int(raw_tracklet.GetPID())
                e_id = int(raw_tracklet.GetEID())
                hits = []

                for hit in self._root_tracklet_hits(raw_tracklet, event_entry["hits"]):
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
                        detector_side=detector_side,
                        volume_id=int(vid),
                        volume_name=vname,
                    )
                    hits.append(hit_obj)

                tracklet = Tracklet(
                    tracklet_id=tracklet_counter,
                    particle_id=particle_id,
                    e_id=e_id,
                    hits=hits
                )

                tracklet.extra_info["pattern_reco"] = pattern_idx
                tracklets.append(tracklet)

                pattern_tracklet_ids.append(tracklet_counter)

                tracklet_counter += 1

            patterns_reco[pattern_idx] = pattern_tracklet_ids

        truth_source = reference_truth_entry if reference_truth_entry is not None else event_entry

        n_patterns_truth = 0
        particles_counter = Counter()
        patterns_truth = {}

        for pattern_idx, pattern in enumerate(truth_source["patterns"]):
            n_patterns_truth += 1
            indices = pattern.GetTrackletIndices()
            pattern_tracklet_ids = []

            for index in indices:
                index = int(index)
                tracklet = truth_source["tracklets"][index]
                particle_id = int(tracklet.GetPID())

                particles_counter[particle_id] += 1
                pattern_tracklet_ids.append(index)

            patterns_truth[pattern_idx] = pattern_tracklet_ids

        result_info = {
            "n_patterns_truth": n_patterns_truth,
            "particles_in_event_truth": particles_counter,
            "patterns_truth": patterns_truth,
            "patterns_reco": patterns_reco
        }

        return tracklets, result_info
