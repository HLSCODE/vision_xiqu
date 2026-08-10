"""HSV segmentation and contour measurements for orange-yellow objects."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import VisionConfig
from vision.models import DetectedObject, DetectionResult


class HSVObjectDetector:
    """Detect static orange/yellow objects using configurable HSV thresholds."""

    def __init__(self, config: VisionConfig, intrinsic_path: Path | None = None, enable_undistort: bool = False) -> None:
        self.config = config
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs: np.ndarray | None = None
        if enable_undistort and intrinsic_path is not None and intrinsic_path.exists():
            # 内参缺失时保持原图处理，保证畸变校正是可选能力。
            with np.load(intrinsic_path) as data:
                self._camera_matrix = data["camera_matrix"]
                self._dist_coeffs = data["dist_coeffs"]

    @property
    def undistortion_enabled(self) -> bool:
        return self._camera_matrix is not None and self._dist_coeffs is not None

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Optionally undistort a BGR frame before colour segmentation."""
        if not self.undistortion_enabled:
            return frame_bgr
        return cv2.undistort(frame_bgr, self._camera_matrix, self._dist_coeffs)

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        """Return all contours above min_area_px; size filtering is caller-owned."""
        if frame_bgr is None or frame_bgr.ndim != 3:
            raise ValueError("Expected a non-empty BGR image with three channels")
        frame = self.preprocess(frame_bgr)
        # OpenCV 相机帧通常是 BGR；阈值统一在 HSV 空间配置，便于现场调参。
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.asarray(self.config.hsv_lower, dtype=np.uint8)
        upper = np.asarray(self.config.hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel_size = max(1, int(self.config.morphology_kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # 先开运算剔除零散噪点，再闭运算填补同一目标内的小孔洞。
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects: list[DetectedObject] = []
        for contour in contours:
            area_px = float(cv2.contourArea(contour))
            if area_px < self.config.min_area_px:
                continue
            rect = cv2.minAreaRect(contour)
            (u, v), (width_px, height_px), _ = rect
            # 首版只依据像素尺寸；不在这里推算实际毫米尺寸或三维坐标。
            objects.append(
                DetectedObject(
                    center=np.array([u, v], dtype=np.float64),
                    contour=contour,
                    area_px=area_px,
                    width_px=float(width_px),
                    height_px=float(height_px),
                    size_px=max(float(width_px), float(height_px)),
                )
            )
        objects.sort(key=lambda item: (item.center[1], item.center[0]))
        for index, obj in enumerate(objects):
            obj.index = index
        return DetectionResult(objects=objects, mask=mask)

    def valid_objects(self, objects: list[DetectedObject]) -> list[DetectedObject]:
        """Return objects eligible for picking at the fixed observe height only."""
        # 此筛选仅供 GLOBAL_DETECT 使用；IBVS 期间必须持续跟踪已锁定目标。
        return [obj for obj in objects if obj.size_px < self.config.size_threshold_px]
