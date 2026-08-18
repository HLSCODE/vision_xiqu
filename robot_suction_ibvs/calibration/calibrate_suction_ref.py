"""Capture the visible target centre when the suction axis is directly aligned.

Before pressing A, manually place the suction axis directly above one stationary
target at the normal observation height and pose. The target must remain visible.
The captured centre becomes ``suction_ref.npy`` and is the direct IBVS goal.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from app.calibration_data import (
    file_sha256,
    load_calibration_array,
    make_metadata,
    save_calibration_array,
    validate_metadata_for_config,
)
from app.config import load_config
from camera.opencv_camera import OpenCVCamera
from vision.detector import RGBObjectDetector


def centre_and_jitter(samples: deque[np.ndarray]) -> tuple[np.ndarray, float]:
    points = np.stack(samples).astype(np.float64)
    centre = np.mean(points, axis=0)
    return centre, float(np.max(np.linalg.norm(points - centre, axis=1)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture the direct IBVS suction-axis reference pixel")
    parser.add_argument("--config", default="robot_suction_ibvs/config.yaml")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--max-jitter-px", type=float, default=3.0)
    args = parser.parse_args()
    if args.frames < 3 or args.max_jitter_px <= 0:
        parser.error("--frames must be at least 3 and --max-jitter-px must be positive")

    config = load_config(args.config)
    servo_path = config.path(config.ibvs.servo_A_path)
    reference_path = config.path(config.suction.suction_ref_path)
    matrix, servo_metadata = load_calibration_array(servo_path, "servo_A")
    if matrix.shape != (2, 2):
        raise ValueError("servo_A must have shape (2, 2)")
    validate_metadata_for_config(servo_metadata, config)

    detector = RGBObjectDetector(
        config.vision,
        config.path(config.camera.intrinsic_path),
        config.camera.enable_undistort,
        expected_image_size=(config.camera.width, config.camera.height),
    )
    camera = OpenCVCamera(config.camera)
    samples: deque[np.ndarray] = deque(maxlen=args.frames)
    print("Place the suction axis directly above one visible stationary target.")
    print("Wait for a stable window, press A to save; press Q or Esc to cancel.")

    camera.open()
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                raise RuntimeError("Camera stream failed")
            result = detector.detect(frame)
            display = detector.preprocess(frame).copy()
            valid = detector.valid_objects(result.objects)
            target = valid[0] if len(valid) == 1 else None
            if target is None:
                samples.clear()
            else:
                samples.append(target.center.copy())
                cv2.drawMarker(
                    display,
                    tuple(np.round(target.center).astype(int)),
                    (0, 255, 255),
                    cv2.MARKER_CROSS,
                    26,
                    2,
                )

            jitter_text = "--"
            if samples:
                _, jitter = centre_and_jitter(samples)
                jitter_text = f"{jitter:.2f}"
            cv2.putText(display, "SUCTION AXIS ALIGNED: press A", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
            cv2.putText(
                display,
                f"valid targets={len(valid)} samples={len(samples)}/{args.frames} jitter={jitter_text}px",
                (12, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 0) if target is not None else (0, 0, 255),
                2,
            )
            cv2.imshow("Suction-axis reference calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 1
            if key != ord("a"):
                continue
            if len(samples) < args.frames:
                print(f"Need {args.frames - len(samples)} more consecutive valid frames")
                continue

            centre, jitter = centre_and_jitter(samples)
            if jitter > args.max_jitter_px:
                print(f"Target is not stable: jitter={jitter:.3f}px > {args.max_jitter_px:.3f}px")
                samples.clear()
                continue

            metadata = make_metadata(
                config,
                "suction_ref",
                servo_metadata.work_frame_name,
                servo_metadata.work_frame_pose_m_rad,
                servo_A_sha256=file_sha256(servo_path),
            )
            save_calibration_array(reference_path, centre, metadata)
            print(f"Saved direct suction reference [u,v]={centre.round(3).tolist()}px to {reference_path}")
            return 0
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
