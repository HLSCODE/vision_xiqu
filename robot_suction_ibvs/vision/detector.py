"""RGB segmentation and contour measurements for configured target colours."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import VisionConfig
from vision.models import DetectedObject, DetectionResult


class RGBObjectDetector:
    """Detect static objects using the RGB range configured in ``config.yaml``."""

    def __init__(
        self,
        config: VisionConfig,
        intrinsic_path: Path | None = None,
        enable_undistort: bool = False,
        expected_image_size: tuple[int, int] | None = None,
        processing_image_size: tuple[int, int] | None = None,
    ) -> None:
        self.config = config
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs: np.ndarray | None = None
        if not enable_undistort:
            return
        if intrinsic_path is None or not intrinsic_path.exists():
            raise FileNotFoundError(f"畸变校正已启用，但相机内参文件不存在：{intrinsic_path}")
        with np.load(intrinsic_path, allow_pickle=False) as data:
            required = {"camera_matrix", "dist_coeffs", "image_size"}
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"相机内参文件缺少字段：{', '.join(missing)}")
            matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
            dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64)
            calibrated_size_values = np.asarray(data["image_size"]).reshape(-1)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("camera_matrix must be a finite 3x3 matrix")
        if dist_coeffs.size < 4 or not np.all(np.isfinite(dist_coeffs)):
            raise ValueError("dist_coeffs must contain at least four finite coefficients")
        if calibrated_size_values.size != 2:
            raise ValueError("camera intrinsic image_size must contain [width, height]")
        calibrated_size = tuple(int(value) for value in calibrated_size_values)
        if any(value <= 0 for value in calibrated_size):
            raise ValueError(f"camera intrinsic image_size is invalid: {calibrated_size!r}")
        source_size = calibrated_size if expected_image_size is None else tuple(expected_image_size)
        if source_size != calibrated_size:
            raise ValueError(
                f"相机内参分辨率 {calibrated_size} 与当前采集分辨率 {source_size} 不一致"
            )
        target_size = source_size if processing_image_size is None else tuple(processing_image_size)
        if any(value <= 0 for value in target_size):
            raise ValueError(f"processing_image_size is invalid: {target_size!r}")
        scale_x = target_size[0] / source_size[0]
        scale_y = target_size[1] / source_size[1]
        scaled_matrix = matrix.copy()
        scaled_matrix[0, :] *= scale_x
        scaled_matrix[1, :] *= scale_y
        self._camera_matrix = scaled_matrix
        self._dist_coeffs = dist_coeffs

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
        # OpenCV 相机帧通常为 BGR；先显式转换后，配置始终按 [R, G, B] 理解。
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lower = np.asarray(self.config.rgb_lower, dtype=np.uint8)
        upper = np.asarray(self.config.rgb_upper, dtype=np.uint8)
        mask = cv2.inRange(rgb, lower, upper)
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
