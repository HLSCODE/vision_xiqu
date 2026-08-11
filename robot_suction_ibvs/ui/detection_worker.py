"""相机预览使用的后台目标检测线程。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from app.config import AppConfig, VisionConfig
from vision.detector import HSVObjectDetector
from vision.models import DetectedObject


@dataclass(frozen=True, slots=True)
class PreviewDetection:
    """一帧预览图像的检测结果，坐标均属于预览图像。"""

    frame_bgr: np.ndarray
    objects: list[DetectedObject]
    valid_indices: frozenset[int]
    frame_width: int
    frame_height: int

    @property
    def valid_count(self) -> int:
        return len(self.valid_indices)


class PreviewDetectionWorker(QThread):
    """在后台执行 HSV 检测；处理不过来时主动丢弃旧预览帧。"""

    detections_ready = pyqtSignal(object)
    detection_failed = pyqtSignal(str)

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self._vision_config = config.vision
        self._intrinsic_path = config.path(config.camera.intrinsic_path)
        self._enable_undistort = config.camera.enable_undistort
        self._condition = threading.Condition()
        self._pending: tuple[np.ndarray, tuple[int, int]] | None = None
        self._stop_requested = False
        self._last_error: str | None = None

    def submit_frame(self, frame_bgr: np.ndarray, source_resolution: tuple[int, int]) -> None:
        """提交最新帧；若上一帧尚未处理，它会被这一帧替换。"""
        if not self.isRunning():
            return
        with self._condition:
            # OpenCV 检测器不会修改输入帧，因此无需再次复制预览图像。
            self._pending = (frame_bgr, source_resolution)
            self._condition.notify()

    def stop(self) -> None:
        """唤醒等待中的线程并请求安全退出。"""
        with self._condition:
            self._stop_requested = True
            self._pending = None
            self._condition.notify_all()

    def run(self) -> None:
        detector: HSVObjectDetector | None = None
        detector_key: tuple[int, int, int, int] | None = None

        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop_requested or self._pending is not None
                )
                if self._stop_requested:
                    return
                pending = self._pending
                self._pending = None

            if pending is None:
                continue
            frame, source_resolution = pending
            try:
                frame_height, frame_width = frame.shape[:2]
                source_width, source_height = source_resolution
                key = (frame_width, frame_height, source_width, source_height)
                if detector is None or key != detector_key:
                    detector = HSVObjectDetector(
                        self._scaled_config(frame_width, frame_height, source_width, source_height),
                        intrinsic_path=self._intrinsic_path,
                        enable_undistort=self._enable_undistort,
                        expected_image_size=(source_width, source_height),
                        processing_image_size=(frame_width, frame_height),
                    )
                    detector_key = key

                result = detector.detect(frame)
                processed_frame = detector.preprocess(frame)
                valid_indices = frozenset(
                    obj.index for obj in detector.valid_objects(result.objects)
                )
                self.detections_ready.emit(
                    PreviewDetection(
                        frame_bgr=processed_frame,
                        objects=result.objects,
                        valid_indices=valid_indices,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                )
                self._last_error = None
            except Exception as exc:
                message = str(exc)
                # 同一种错误只上报一次，避免运行日志被每帧重复刷屏。
                if message != self._last_error:
                    self._last_error = message
                    self.detection_failed.emit(message)

    def _scaled_config(
        self,
        frame_width: int,
        frame_height: int,
        source_width: int,
        source_height: int,
    ) -> VisionConfig:
        """把原始分辨率下的面积和尺寸阈值换算到预览分辨率。"""
        if source_width <= 0 or source_height <= 0:
            scale = 1.0
        else:
            scale = min(frame_width / source_width, frame_height / source_height)
        scale = min(max(scale, 1e-6), 1.0)

        kernel = max(1, int(round(self._vision_config.morphology_kernel * scale)))
        # 奇数核具有明确中心，OpenCV 形态学操作更稳定。
        if kernel % 2 == 0:
            kernel += 1
        return replace(
            self._vision_config,
            min_area_px=max(1.0, self._vision_config.min_area_px * scale * scale),
            morphology_kernel=kernel,
            size_threshold_px=max(1.0, self._vision_config.size_threshold_px * scale),
        )
