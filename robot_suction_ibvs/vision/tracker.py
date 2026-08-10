"""Deliberately simple nearest-neighbour locked-target association."""

from __future__ import annotations

import numpy as np

from vision.models import DetectedObject, TrackingResult


class NearestNeighborTracker:
    """Associate only the target locked before IBVS; no persistent global IDs."""

    def __init__(self, max_distance_px: float) -> None:
        self.max_distance_px = max_distance_px

    def select_nearest_reference(self, objects: list[DetectedObject], reference_px: np.ndarray) -> DetectedObject | None:
        """Select the globally valid candidate closest to the suction reference point."""
        if not objects:
            return None
        return min(objects, key=lambda item: float(np.linalg.norm(item.center - reference_px)))

    def associate(self, last_center_px: np.ndarray, objects: list[DetectedObject]) -> TrackingResult:
        """Find the nearest current contour to the last locked target centre."""
        if not objects:
            return TrackingResult(target=None, distance_px=None)
        candidate = min(objects, key=lambda item: float(np.linalg.norm(item.center - last_center_px)))
        distance = float(np.linalg.norm(candidate.center - last_center_px))
        if distance > self.max_distance_px:
            return TrackingResult(target=None, distance_px=distance)
        return TrackingResult(target=candidate, distance_px=distance)
