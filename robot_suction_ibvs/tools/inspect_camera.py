"""Quickly verify camera device selection, stream stability, and actual resolution."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from app.config import load_config
from camera.opencv_camera import OpenCVCamera


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect an OpenCV USB camera, file, or RTSP stream")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", default=None, help="Optional device id, file, or RTSP URL")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.source is not None:
        source = int(args.source) if args.source.isdigit() else args.source
        config = type(config)(**{**config.__dict__, "camera": type(config.camera)(**{**config.camera.__dict__, "device_id": source, "video_path": None})})
    camera = OpenCVCamera(config.camera)
    camera.open()
    print(f"Opened camera. Actual resolution: {camera.get_resolution()}. Press Q/Esc to exit.")
    previous = time.monotonic()
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                print("Frame read failed")
                return 1
            now = time.monotonic()
            fps = 1.0 / max(now - previous, 1e-6)
            previous = now
            cv2.putText(frame, f"{frame.shape[1]}x{frame.shape[0]}  {fps:.1f} FPS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Camera inspection", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return 0
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
