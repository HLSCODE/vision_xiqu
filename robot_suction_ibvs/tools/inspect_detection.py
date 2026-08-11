"""Visualize HSV contours, pixel sizes, and observation-only size eligibility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from app.config import load_config
from camera.opencv_camera import OpenCVCamera
from vision.detector import HSVObjectDetector
from vision.visualization import draw_detection_overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect HSV detection and size_px filtering")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    camera = OpenCVCamera(config.camera)
    detector = HSVObjectDetector(config.vision, config.path(config.camera.intrinsic_path), config.camera.enable_undistort)
    camera.open()
    print("绿色表示尺寸合格，橙色表示尺寸超限。按 Q/Esc 退出。")
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                return 1
            result = detector.detect(frame)
            valid = detector.valid_objects(result.objects)
            valid_indices = frozenset(obj.index for obj in valid)
            overlay = draw_detection_overlay(frame, result.objects, valid_indices)
            cv2.imshow("Detection inspection", overlay)
            cv2.imshow("HSV mask", result.mask)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return 0
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
