"""Interactive HSV threshold tuner that saves values as portable JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from app.config import load_config
from camera.opencv_camera import OpenCVCamera
from vision.detector import HSVObjectDetector


def nothing(_: int) -> None:
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune HSV segmentation using OpenCV trackbars")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", default=None, help="Optional device id, video path, or RTSP URL")
    parser.add_argument("--output", default="data/hsv_tuned.json", help="JSON file written after S is pressed")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.source is not None:
        device = int(args.source) if args.source.isdigit() else args.source
        camera_cfg = type(config.camera)(**{**config.camera.__dict__, "device_id": device, "video_path": None})
    else:
        camera_cfg = config.camera
    camera = OpenCVCamera(camera_cfg)
    preprocessor = HSVObjectDetector(
        config.vision,
        config.path(config.camera.intrinsic_path),
        config.camera.enable_undistort,
        expected_image_size=(camera_cfg.width, camera_cfg.height),
    )
    window = "HSV controls"
    cv2.namedWindow(window)
    defaults = (*config.vision.hsv_lower, *config.vision.hsv_upper)
    labels = ("H_min", "S_min", "V_min", "H_max", "S_max", "V_max")
    limits = (179, 255, 255, 179, 255, 255)
    for label, value, limit in zip(labels, defaults, limits):
        cv2.createTrackbar(label, window, int(value), limit, nothing)
    camera.open()
    print("Adjust sliders. Press S to save JSON, Q/Esc to exit.")
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                raise RuntimeError("Camera/video frame read failed")
            frame = preprocessor.preprocess(frame)
            values = [cv2.getTrackbarPos(label, window) for label in labels]
            lower, upper = np.array(values[:3], dtype=np.uint8), np.array(values[3:], dtype=np.uint8)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            cv2.imshow("BGR", frame)
            cv2.imshow("Mask", mask)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord("s"):
                output = config.path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("w", encoding="utf-8") as handle:
                    json.dump({"hsv_lower": values[:3], "hsv_upper": values[3:]}, handle, indent=2)
                print(f"Saved {output}. Copy values into vision.hsv_lower / vision.hsv_upper in config.yaml.")
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
