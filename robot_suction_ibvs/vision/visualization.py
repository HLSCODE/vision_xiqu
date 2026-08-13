"""PyQt 相机预览使用的 OpenCV 目标标记。"""

from __future__ import annotations

import cv2
import numpy as np

from vision.models import DetectedObject


def draw_detection_overlay(
    frame_bgr: np.ndarray,
    objects: list[DetectedObject],
    valid_indices: frozenset[int] | set[int],
    source_resolution: tuple[int, int] | None = None,
) -> np.ndarray:
    """绘制目标轮廓和原始采集图像中的 ``(u, v)`` 坐标标签。"""
    canvas = frame_bgr.copy()
    image_height, image_width = canvas.shape[:2]
    source_width, source_height = source_resolution or (image_width, image_height)
    scale_to_source_x = source_width / image_width
    scale_to_source_y = source_height / image_height
    base_size = min(image_width, image_height)
    thickness = max(2, int(round(base_size / 450)))
    font_scale = min(0.7, max(0.45, base_size / 1200))

    for obj in objects:
        is_valid = obj.index in valid_indices
        color = (55, 210, 90) if is_valid else (0, 165, 255)
        cv2.drawContours(canvas, [obj.contour], -1, color, thickness)

        center = tuple(np.round(obj.center).astype(int))
        source_u = int(round(float(obj.center[0]) * scale_to_source_x))
        source_v = int(round(float(obj.center[1]) * scale_to_source_y))
        status = "OK" if is_valid else "LARGE"
        label = f"#{obj.index + 1} {status} (u={source_u}, v={source_v})"
        (label_width, label_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        label_x = min(max(center[0] + 8, 4), max(4, image_width - label_width - 6))
        label_y = min(max(center[1] - 8, 20), max(20, image_height - 6))
        # 黑色描边使标签在浅色容器和高反光表面上仍然清晰。
        cv2.putText(
            canvas,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (20, 20, 20),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return canvas
