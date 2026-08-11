"""PyQt 相机预览使用的 OpenCV 目标标记。"""

from __future__ import annotations

import cv2
import numpy as np

from vision.models import DetectedObject


def draw_detection_overlay(
    frame_bgr: np.ndarray,
    objects: list[DetectedObject],
    valid_indices: frozenset[int] | set[int],
) -> np.ndarray:
    """绘制预览检测轮廓、中心点、编号和像素尺寸。"""
    canvas = frame_bgr.copy()
    image_height, image_width = canvas.shape[:2]
    base_size = min(image_width, image_height)
    thickness = max(2, int(round(base_size / 450)))
    font_scale = min(0.7, max(0.45, base_size / 1200))

    for obj in objects:
        is_valid = obj.index in valid_indices
        color = (55, 210, 90) if is_valid else (0, 165, 255)
        cv2.drawContours(canvas, [obj.contour], -1, color, thickness)

        center = tuple(np.round(obj.center).astype(int))
        cv2.drawMarker(
            canvas,
            center,
            color,
            cv2.MARKER_CROSS,
            max(12, thickness * 6),
            thickness,
        )
        status = "OK" if is_valid else "LARGE"
        label = f"#{obj.index + 1} {status} {obj.size_px:.1f}px"
        label_x = min(max(center[0] + 8, 4), max(4, image_width - 190))
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
