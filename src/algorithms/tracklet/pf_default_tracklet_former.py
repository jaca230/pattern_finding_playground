from __future__ import annotations

from collections import Counter
from typing import Any, Optional
import math

from algorithms.tracklet.tracklet_former import TrackletFormer
from models.hit import Hit
from models.point_3d import Point3D
from models.tracklet import Tracklet


class PFDefaultTrackletFormer(TrackletFormer):
    """Python analogue of Reco's PFDefaultTrackletFormer.

    The C++ tool wraps each reconstructed PIRECTracklet and attaches the
    corresponding PIRECTrackletFitResult through GetFitResultIndex(). This
    version does the same, while converting the display-friendly pieces into
    playground Tracklet/Hit objects.
    """

    def form_tracklets(
        self,
        reco_entry: Any,
        geo: Any,
        storage: Optional[Any] = None,
        truth_entry: Optional[Any] = None,
    ) -> tuple[list[Tracklet], dict]:
        if "fit_results" not in reco_entry:
            raise KeyError(
                "PFDefaultTrackletFormer requires reco_entry['fit_results']. "
                "Construct RecoDataFile with fit_result_prefix='_Event_fitres'."
            )

        tracklets: list[Tracklet] = []
        reco_particles_counter: Counter[int] = Counter()

        for tracklet_id, raw_tracklet in enumerate(reco_entry["tracklets"]):
            particle_id = int(raw_tracklet.GetPID())
            reco_particles_counter[particle_id] += 1

            tracklet = Tracklet(
                tracklet_id=tracklet_id,
                particle_id=particle_id,
                e_id=int(raw_tracklet.GetEID()),
                hits=self._build_hits(raw_tracklet, reco_entry["hits"], geo),
            )

            fit_result = self._fit_result_for_tracklet(raw_tracklet, reco_entry["fit_results"])
            if fit_result is not None:
                self._attach_fit_result(tracklet, fit_result)

            tracklet.extra_info["raw_tracklet"] = raw_tracklet
            tracklets.append(tracklet)

        result_info = {
            "input_tracklets_count": len(tracklets),
            "input_tracklet_ptrs_received": int(reco_entry["tracklets"].size()),
            "particles_in_event_reco": reco_particles_counter,
        }

        if truth_entry is not None:
            result_info.update(self._truth_pattern_info(truth_entry))

        return tracklets, result_info

    def _build_hits(self, raw_tracklet: Any, hit_vector: Any, geo: Any) -> list[Hit]:
        hits: list[Hit] = []

        for hit in self._root_tracklet_hits(raw_tracklet, hit_vector):
            vid = hit.GetVID()
            volume_name = geo.GetVolumeName(vid).Data()

            if "atar" not in volume_name:
                continue

            detector_side = self._detector_side(volume_name)
            hits.append(
                Hit(
                    z=geo.GetZ(vid) + 0.07,
                    x=geo.GetX(vid),
                    y=geo.GetY(vid),
                    time=hit.GetObservedTime(),
                    energy=hit.GetObservedEdep(),
                    particle_id=hit.GetPID(),
                    detector_side=detector_side,
                )
            )

        return hits

    def _detector_side(self, volume_name: str) -> str | None:
        if len(volume_name) <= 11:
            return None
        if volume_name[11] == "f":
            return "front"
        if volume_name[11] == "b":
            return "back"
        return None

    def _fit_result_for_tracklet(self, raw_tracklet: Any, fit_results: Any) -> Any | None:
        fit_index = int(raw_tracklet.GetFitResultIndex())
        if fit_index < 0 or fit_index >= int(fit_results.size()):
            return None
        return fit_results[fit_index]

    def _attach_fit_result(self, tracklet: Tracklet, fit_result: Any) -> None:
        tracklet.extra_info["raw_fit_result"] = fit_result
        tracklet.extra_info["fit_endpoints"] = self._extract_fit_endpoints(fit_result)

        start = fit_result.GetStartPoint()
        stop = fit_result.GetStopPoint()
        if start and stop:
            start_pos = start.GetPosition()
            stop_pos = stop.GetPosition()
            tracklet.set_endpoints(
                Point3D(
                    self._sanitize_coordinate(start_pos.X()),
                    self._sanitize_coordinate(start_pos.Y()),
                    self._sanitize_coordinate(start_pos.Z()),
                ),
                Point3D(
                    self._sanitize_coordinate(stop_pos.X()),
                    self._sanitize_coordinate(stop_pos.Y()),
                    self._sanitize_coordinate(stop_pos.Z()),
                ),
            )

    def _extract_fit_endpoints(self, fit_result: Any) -> list[dict[str, Any]]:
        endpoints: list[dict[str, Any]] = []
        for endpoint_index, endpoint in enumerate(fit_result.GetEndpoints()):
            position = endpoint.GetPosition()
            heading = endpoint.GetLinearHeading()
            endpoints.append(
                {
                    "endpoint_index": endpoint_index,
                    "position": (
                        float(position.X()),
                        float(position.Y()),
                        float(position.Z()),
                    ),
                    "linear_heading": (
                        float(heading.X()),
                        float(heading.Y()),
                        float(heading.Z()),
                    ),
                    "vid": int(endpoint.GetVID()),
                    "endpointness": float(endpoint.GetEndpointness()),
                }
            )
        return endpoints

    def _sanitize_coordinate(self, value: float) -> float:
        if math.isnan(value) or math.isinf(value) or abs(value) > 1.0e6:
            return 0.0
        return float(value)

    def _truth_pattern_info(self, truth_entry: Any) -> dict[str, Any]:
        particles_counter: Counter[int] = Counter()
        patterns_truth: dict[int, list[int]] = {}

        for pattern_idx, pattern in enumerate(truth_entry["patterns"]):
            pattern_tracklet_ids: list[int] = []
            for index in pattern.GetTrackletIndices():
                tracklet_index = int(index)
                tracklet = truth_entry["tracklets"][tracklet_index]
                particles_counter[int(tracklet.GetPID())] += 1
                pattern_tracklet_ids.append(tracklet_index)
            patterns_truth[pattern_idx] = pattern_tracklet_ids

        return {
            "n_patterns_truth": len(patterns_truth),
            "particles_in_event_truth": particles_counter,
            "patterns_truth": patterns_truth,
        }
