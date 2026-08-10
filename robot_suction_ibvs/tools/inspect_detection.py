"""Visualize HSV contours, pixel sizes, and observation-only size eligibility."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from app.config import load_config
from app.state import SystemState
from camera.opencv_camera import OpenCVCamera
from vision.detector import HSVObjectDetector
from vision.visualization import draw_debug_overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect HSV detection and size_px filtering")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    camera = OpenCVCamera(config.camera)
    detector = HSVObjectDetector(config.vision, config.path(config.camera.intrinsic_path), config.camera.enable_undistort)
    camera.open()
    reference = np.array([config.camera.width / 2, config.camera.height / 2], dtype=np.float64)
    previous = time.monotonic()
    print("Showing contours. VALID means size_px < vision.size_threshold_px. Press Q/Esc to exit.")
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                return 1
            result = detector.detect(frame)
            valid = detector.valid_objects(result.objects)
            now = time.monotonic()
            overlay = draw_debug_overlay(frame, result.objects, valid, reference, SystemState.GLOBAL_DETECT, fps=1 / max(now - previous, 1e-6))
            previous = now
            cv2.imshow("Detection inspection", overlay)
            cv2.imshow("HSV mask", result.mask)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return 0
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
