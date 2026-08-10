"""Fit the local 2x2 pixel/mm IBVS interaction matrix at observe height."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.config import load_config
from camera.opencv_camera import OpenCVCamera
from control.safety import SafetyManager
from robot.real_robot_template import RealRobot
from vision.detector import HSVObjectDetector


def average_target_center(camera, detector: HSVObjectDetector, frames: int, expected: np.ndarray | None = None) -> np.ndarray:
    """Collect several detections and return the target centre averaged in px."""
    centres: list[np.ndarray] = []
    for _ in range(frames * 3):
        frame = camera.get_frame()
        if frame is None:
            continue
        objects = detector.detect(frame).objects
        if not objects:
            continue
        target = min(objects, key=lambda obj: float(np.linalg.norm(obj.center - expected))) if expected is not None else objects[0]
        centres.append(target.center)
        if len(centres) >= frames:
            break
    if len(centres) < frames:
        raise RuntimeError(f"Only collected {len(centres)}/{frames} valid target observations")
    return np.mean(np.stack(centres), axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate local 2D IBVS pixel/mm matrix A")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--frames", type=int, default=10, help="Frames averaged after every jog")
    parser.add_argument("--output", default=None, help="Defaults to ibvs.servo_A_path in config")
    args = parser.parse_args()
    config = load_config(args.config)
    logger = logging.getLogger("calibrate_servo_xy")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    camera = OpenCVCamera(config.camera)
    robot = RealRobot()  # TODO adapter must be implemented before real calibration.
    detector = HSVObjectDetector(config.vision, config.path(config.camera.intrinsic_path), config.camera.enable_undistort)
    safety = SafetyManager(robot, config.safety)
    output = config.path(args.output or config.ibvs.servo_A_path)
    offsets_mm = np.array([[-2, 0], [-1, 0], [1, 0], [2, 0], [0, -2], [0, -1], [0, 1], [0, 2]], dtype=np.float64)
    camera.open()
    robot.connect()
    try:
        safety.move_to_observe_pose()
        baseline = average_target_center(camera, detector, args.frames)
        pixel_deltas: list[np.ndarray] = []
        print("Move target setup complete. Sampling commanded XY increments at fixed observation height.")
        for dx_mm, dy_mm in offsets_mm:
            safety.move_xy_relative(float(dx_mm), float(dy_mm))
            shifted = average_target_center(camera, detector, args.frames, baseline)
            pixel_deltas.append(shifted - baseline)
            safety.move_xy_relative(float(-dx_mm), float(-dy_mm))
            print(f"dXY=({dx_mm:+.1f}, {dy_mm:+.1f}) mm -> dUV={(shifted - baseline).round(3).tolist()} px")
        coefficients, residuals, rank, singular_values = np.linalg.lstsq(offsets_mm, np.stack(pixel_deltas), rcond=None)
        matrix = coefficients.T  # [du,dv]^T = A_px_per_mm @ [dX,dY]^T
        condition = float(np.linalg.cond(matrix))
        residual = float(residuals.sum()) if residuals.size else 0.0
        print("A (px/mm) =\n", matrix)
        print(f"least-squares residual={residual:.6f}, rank={rank}, condition={condition:.3f}, singular_values={singular_values}")
        if not np.isfinite(condition) or condition > 1e5:
            print("WARNING: matrix is singular or poorly conditioned; do not use it for real motion.")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, matrix)
        print(f"Saved {output}")
        return 0
    finally:
        safety.stop_all()
        robot.disconnect()
        camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
