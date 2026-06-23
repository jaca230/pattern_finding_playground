from __future__ import annotations

from collections import Counter
from typing import Any, Optional
import math

from algorithms.registry import register_algorithm
from algorithms.tracklet.tracklet_former import TrackletFormer
from models.hit import Hit
from models.tracklet import Tracklet


@register_algorithm(
    "tracklet",
    name="default",
    description="Wrap the active ATAR tracklet collections, attach fit results, and derive display-safe endpoints.",
    parameters={},
    example="DefaultTrackletFormer()",
)
class DefaultTrackletFormer(TrackletFormer):
    """Python analogue of Reco's default tracklet wrapping for the active event view."""

    ENDPOINT_Z_MARGIN_MM = 1.0
    ENDPOINT_COORD_MARGIN_MM = 1.0

    def form_tracklets(
        self,
        event_entry: Any,
        geo: Any,
        storage: Optional[Any] = None,
        reference_truth_entry: Optional[Any] = None,
    ) -> tuple[list[Tracklet], dict]:
        if "fit_results" not in event_entry:
            raise KeyError("DefaultTrackletFormer requires fit_results in the active event entry.")

        tracklets: list[Tracklet] = []
        reco_particles_counter: Counter[int] = Counter()

        for tracklet_id, raw_tracklet in enumerate(event_entry["tracklets"]):
            particle_id = int(raw_tracklet.GetPID())
            reco_particles_counter[particle_id] += 1

            tracklet = Tracklet(
                tracklet_id=tracklet_id,
                particle_id=particle_id,
                e_id=int(raw_tracklet.GetEID()),
                hits=self._build_hits(raw_tracklet, event_entry["hits"], geo),
            )

            fit_result = self._fit_result_for_tracklet(raw_tracklet, event_entry["fit_results"])
            if fit_result is not None:
                self._attach_fit_result(tracklet, fit_result)

            tracklet.extra_info["raw_tracklet"] = raw_tracklet
            tracklets.append(tracklet)

        result_info = {
            "input_tracklets_count": len(tracklets),
            "input_tracklet_ptrs_received": int(event_entry["tracklets"].size()),
            "particles_in_event_reco": reco_particles_counter,
        }

        if reference_truth_entry is not None:
            result_info.update(self._truth_pattern_info(reference_truth_entry))

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
                    volume_id=int(vid),
                    volume_name=volume_name,
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
        fit_endpoints = self._extract_fit_endpoints(fit_result)
        tracklet.extra_info["fit_endpoints"] = fit_endpoints
        tracklet.extra_info["display_endpoints"] = self._display_endpoints_by_plane(tracklet, fit_endpoints)

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

    def _display_endpoints_by_plane(
        self,
        tracklet: Tracklet,
        fit_endpoints: list[dict[str, Any]],
    ) -> dict[str, list[tuple[float, float]]]:
        plane_points = {"xz": [], "yz": []}

        for endpoint in fit_endpoints:
            x, y, z = endpoint["position"]
            if self._endpoint_in_hit_envelope(tracklet, "xz", z, x):
                plane_points["xz"].append((float(z), float(x)))
            if self._endpoint_in_hit_envelope(tracklet, "yz", z, y):
                plane_points["yz"].append((float(z), float(y)))

        for plane in plane_points:
            unique_points = sorted(set(plane_points[plane]), key=lambda point: point[0])
            plane_points[plane] = unique_points

        return plane_points

    def _is_display_coordinate(self, value: float) -> bool:
        return not math.isnan(value) and not math.isinf(value) and abs(value) < 1.0e6

    def _endpoint_in_hit_envelope(
        self,
        tracklet: Tracklet,
        plane: str,
        z_value: float,
        coord_value: float,
    ) -> bool:
        if not self._is_display_coordinate(z_value) or not self._is_display_coordinate(coord_value):
            return False

        z_bounds, coord_bounds = self._projected_hit_bounds(tracklet, plane)
        if z_bounds is None or coord_bounds is None:
            return False

        z_low, z_high = z_bounds
        coord_low, coord_high = coord_bounds
        return (
            z_low - self.ENDPOINT_Z_MARGIN_MM <= z_value <= z_high + self.ENDPOINT_Z_MARGIN_MM
            and coord_low - self.ENDPOINT_COORD_MARGIN_MM <= coord_value <= coord_high + self.ENDPOINT_COORD_MARGIN_MM
        )

    def _projected_hit_bounds(
        self,
        tracklet: Tracklet,
        plane: str,
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        expected_side = "front" if plane == "xz" else "back"
        z_values = []
        coord_values = []
        for hit in tracklet.hits:
            if hit.detector_side != expected_side or hit.z is None:
                continue
            coord = hit.x if plane == "xz" else hit.y
            if coord is None:
                continue
            z_values.append(float(hit.z))
            coord_values.append(float(coord))

        if not z_values or not coord_values:
            return None, None
        return (min(z_values), max(z_values)), (min(coord_values), max(coord_values))

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
