"""OpenCV debug overlays for observation and IBVS control."""

from __future__ import annotations

import cv2
import numpy as np

from app.state import SystemState
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


def draw_debug_overlay(
    frame_bgr: np.ndarray,
    objects: list[DetectedObject],
    valid_objects: list[DetectedObject],
    suction_ref_px: np.ndarray,
    state: SystemState,
    target: DetectedObject | None = None,
    error_px: np.ndarray | None = None,
    velocity_mm_s: np.ndarray | None = None,
    fps: float | None = None,
) -> np.ndarray:
    """Return an annotated BGR frame without mutating the capture buffer."""
    canvas = frame_bgr.copy()
    valid_ids = {id(obj) for obj in valid_objects}
    for obj in objects:
        color = (0, 200, 0) if id(obj) in valid_ids else (0, 80, 255)
        cv2.drawContours(canvas, [obj.contour], -1, color, 2)
        point = tuple(np.round(obj.center).astype(int))
        label = f"#{obj.index} size={obj.size_px:.1f}px"
        if id(obj) in valid_ids:
            label += " VALID"
        cv2.putText(canvas, label, (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.circle(canvas, point, 3, color, -1)
    ref = tuple(np.round(suction_ref_px).astype(int))
    cv2.drawMarker(canvas, ref, (255, 255, 0), cv2.MARKER_CROSS, 22, 2)
    cv2.putText(canvas, "SUCTION REF", (ref[0] + 8, ref[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    if target is not None:
        point = tuple(np.round(target.center).astype(int))
        cv2.circle(canvas, point, 18, (255, 0, 255), 2)
        cv2.putText(canvas, "TARGET", (point[0] + 18, point[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
    lines = [f"STATE: {state.name}"]
    if error_px is not None:
        lines.append(f"error: ({error_px[0]:.1f}, {error_px[1]:.1f}) px")
    if velocity_mm_s is not None:
        lines.append(f"vxy: ({velocity_mm_s[0]:.2f}, {velocity_mm_s[1]:.2f}) mm/s")
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")
    for row, text in enumerate(lines):
        cv2.putText(canvas, text, (12, 28 + row * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, text, (12, 28 + row * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
    return canvas
