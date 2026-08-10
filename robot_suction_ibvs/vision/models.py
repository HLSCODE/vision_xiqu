"""Data models shared by detection, tracking, and visualization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DetectedObject:
    """One contour-derived object in image coordinates (px)."""

    center: np.ndarray  # shape (2,), [u, v], px
    contour: np.ndarray
    area_px: float
    width_px: float
    height_px: float
    size_px: float
    index: int = -1


@dataclass(slots=True)
class DetectionResult:
    """Detection output for one BGR frame."""

    objects: list[DetectedObject]
    mask: np.ndarray


@dataclass(slots=True)
class TrackingResult:
    """Nearest-neighbour association output for a locked target."""

    target: DetectedObject | None
    distance_px: float | None
