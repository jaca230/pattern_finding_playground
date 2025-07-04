from deprecated.algorithms.validation.event_validator import EventValidator
from deprecated.models.event_patterns import EventPatterns
from typing import List

class TrackletGroupingValidator(EventValidator):
    def __init__(self, verbose: int = 0):
        self.verbose = verbose

    def validate(self, event: EventPatterns) -> bool:
        """
        Validates that the tracklet IDs in each reconstructed pattern match the ones
        stored in the truth-level `patterns_truth` from the algorithm_info.

        Returns True if valid, False otherwise.
        """
        if "tracklet_algorithm_info" not in event.extra_info:
            raise ValueError("No tracklet algorithm info found in EventPatterns.")

        patterns_truth = event.extra_info["tracklet_algorithm_info"].get("patterns_truth")
        if patterns_truth is None:
            raise ValueError("No 'patterns_truth' found in tracklet algorithm info.")

        reconstructed_patterns = sorted(event.get_patterns(), key=lambda p: p.pattern_id)
        reconstructed_tracklet_sets = [
            {t.tracklet_id for t in pattern.get_unique_tracklets()}
            for pattern in reconstructed_patterns
        ]

        # Convert to list of sets for easy comparison
        truth_tracklet_sets = [
            set(tracklet_ids) for _, tracklet_ids in sorted(patterns_truth.items())
        ]

        if len(truth_tracklet_sets) != len(reconstructed_tracklet_sets):
            if self.verbose > 0:
                print(f"[Validation Error] Pattern count mismatch: "
                      f"{len(truth_tracklet_sets)} (truth) vs {len(reconstructed_tracklet_sets)} (reco)")
            return False

        for i, (truth_set, reco_set) in enumerate(zip(truth_tracklet_sets, reconstructed_tracklet_sets)):
            if truth_set != reco_set:
                if self.verbose > 0:
                    print(f"[Validation Error] Tracklet ID mismatch in pattern {i}:")
                    print(f"  Truth: {sorted(truth_set)}")
                    print(f"  Reco : {sorted(reco_set)}")
                return False

        if self.verbose > 0:
            print("[Validation Success] All patterns validated successfully.")

        return True