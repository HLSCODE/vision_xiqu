"""Record the pixel location of a physically aligned suction axis."""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from app.config import load_config
from camera.opencv_camera import OpenCVCamera
from vision.detector import HSVObjectDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate suction_ref_pixel after manual XY alignment")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--frames", type=int, default=20, help="Valid detection centres to average")
    parser.add_argument("--output", default=None, help="Defaults to suction.suction_ref_path in config")
    args = parser.parse_args()
    config = load_config(args.config)
    camera = OpenCVCamera(config.camera)
    detector = HSVObjectDetector(config.vision, config.path(config.camera.intrinsic_path), config.camera.enable_undistort)
    output = config.path(args.output or config.suction.suction_ref_path)
    centres: deque[np.ndarray] = deque(maxlen=max(1, args.frames))
    camera.open()
    print("Manually jog until the suction axis is exactly above one orange target. Press S to save; Q/Esc to cancel.")
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                raise RuntimeError("Camera stream failed")
            result = detector.detect(frame)
            candidates = detector.valid_objects(result.objects)
            if candidates:
                obj = min(candidates, key=lambda item: item.area_px)
                centres.append(obj.center.copy())
                point = tuple(np.round(obj.center).astype(int))
                cv2.drawMarker(frame, point, (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
                cv2.putText(frame, f"samples {len(centres)}/{centres.maxlen}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Suction reference calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 1
            if key == ord("s"):
                if len(centres) < centres.maxlen:
                    print(f"Need {centres.maxlen - len(centres)} more valid frames")
                    continue
                reference = np.mean(np.stack(centres), axis=0)
                output.parent.mkdir(parents=True, exist_ok=True)
                np.save(output, reference)
                print(f"Saved suction_ref_pixel [u, v] px = {reference.tolist()} to {output}")
                return 0
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
