"""Fit the local 2x2 pixel/mm IBVS interaction matrix at observe height."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.calibration_data import make_metadata, save_calibration_array
from app.config import load_config
from camera.opencv_camera import OpenCVCamera
from control.safety import SafetyManager
from robot.realman_robot import RealRobot
from vision.detector import HSVObjectDetector


def average_target_center(
    camera: OpenCVCamera,
    detector: HSVObjectDetector,
    frames: int,
    max_jitter_px: float,
) -> np.ndarray:
    """Average consecutive frames containing exactly one stable, size-eligible target."""
    centres: list[np.ndarray] = []
    for _ in range(frames * 5):
        frame = camera.get_frame()
        if frame is None:
            centres.clear()
            continue
        objects = detector.valid_objects(detector.detect(frame).objects)
        if len(objects) != 1:
            centres.clear()
            continue
        centres.append(objects[0].center.copy())
        if len(centres) >= frames:
            break
    if len(centres) < frames:
        raise RuntimeError(
            f"Only collected {len(centres)}/{frames} consecutive frames with exactly one valid target"
        )
    points = np.stack(centres).astype(np.float64)
    centre = np.mean(points, axis=0)
    jitter = float(np.max(np.linalg.norm(points - centre, axis=1)))
    if jitter > max_jitter_px:
        raise RuntimeError(
            f"Target jitter {jitter:.3f}px exceeds configured limit {max_jitter_px:.3f}px"
        )
    return centre


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate local 2D IBVS pixel/mm matrix A")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--frames", type=int, default=10, help="Frames averaged after every jog")
    parser.add_argument("--max-jitter-px", type=float, default=3.0)
    parser.add_argument("--max-return-error-px", type=float, default=3.0)
    parser.add_argument("--max-fit-rms-px", type=float, default=2.0)
    parser.add_argument("--output", default=None, help="Defaults to ibvs.servo_A_path in config")
    args = parser.parse_args()
    if args.frames < 3:
        parser.error("--frames must be at least 3")
    if args.max_jitter_px <= 0 or args.max_return_error_px <= 0 or args.max_fit_rms_px <= 0:
        parser.error("jitter, return-error and fit-RMS limits must be positive")
    config = load_config(args.config)
    logger = logging.getLogger("calibrate_servo_xy")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    camera = OpenCVCamera(config.camera)
    robot = RealRobot(config.robot)
    detector = HSVObjectDetector(
        config.vision,
        config.path(config.camera.intrinsic_path),
        config.camera.enable_undistort,
        expected_image_size=(config.camera.width, config.camera.height),
    )
    safety = SafetyManager(robot, config.safety)
    output = config.path(args.output or config.ibvs.servo_A_path)
    offsets_mm = np.array([[-2, 0], [-1, 0], [1, 0], [2, 0], [0, -2], [0, -1], [0, 1], [0, 2]], dtype=np.float64)
    camera.open()
    robot.connect()
    try:
        work_frame_name = robot.get_current_work_frame_name()
        work_frame_pose = robot.get_current_work_frame_pose()
        safety.move_to_observe_pose()
        safety.begin_alignment()
        baseline = average_target_center(camera, detector, args.frames, args.max_jitter_px)
        pixel_deltas: list[np.ndarray] = []
        print(
            "Move target setup complete. Sampling commanded XY increments at fixed observation height "
            f"in work frame {work_frame_name!r}."
        )
        for dx_mm, dy_mm in offsets_mm:
            safety.move_xy_relative(float(dx_mm), float(dy_mm))
            try:
                shifted = average_target_center(camera, detector, args.frames, args.max_jitter_px)
            finally:
                # 即使采样失败，也优先尝试回到本次微移前的位置。
                safety.move_xy_relative(float(-dx_mm), float(-dy_mm))
            delta = shifted - baseline
            returned = average_target_center(camera, detector, args.frames, args.max_jitter_px)
            return_error = float(np.linalg.norm(returned - baseline))
            if return_error > args.max_return_error_px:
                raise RuntimeError(
                    f"Robot/camera did not return to the image baseline: {return_error:.3f}px > "
                    f"{args.max_return_error_px:.3f}px"
                )
            pixel_deltas.append(delta)
            baseline = returned
            print(
                f"dXY=({dx_mm:+.1f}, {dy_mm:+.1f}) mm -> dUV={delta.round(3).tolist()} px, "
                f"return_error={return_error:.3f}px"
            )
        measured_deltas = np.stack(pixel_deltas)
        coefficients, _, rank, singular_values = np.linalg.lstsq(offsets_mm, measured_deltas, rcond=None)
        matrix = coefficients.T  # [du,dv]^T = A_px_per_mm @ [dX,dY]^T
        condition = float(np.linalg.cond(matrix))
        residual_vectors = offsets_mm @ coefficients - measured_deltas
        fit_rms_px = float(np.sqrt(np.mean(np.sum(residual_vectors * residual_vectors, axis=1))))
        print("A (px/mm) =\n", matrix)
        print(
            f"fit_rms={fit_rms_px:.6f}px, rank={rank}, condition={condition:.3f}, "
            f"singular_values={singular_values}"
        )
        if rank < 2:
            raise RuntimeError(f"Calibration fit rank is {rank}; both robot XY axes must be observable")
        if not np.isfinite(condition) or condition > 1e5:
            raise RuntimeError(f"Calibration matrix is singular or poorly conditioned: {condition:.3e}")
        if fit_rms_px > args.max_fit_rms_px:
            raise RuntimeError(
                f"Calibration fit RMS {fit_rms_px:.3f}px exceeds {args.max_fit_rms_px:.3f}px"
            )
        metadata = make_metadata(config, "servo_A", work_frame_name, work_frame_pose)
        save_calibration_array(output, matrix, metadata)
        print(f"Saved {output} and its calibration-context sidecar")
        return 0
    finally:
        try:
            safety.stop_all()
        finally:
            try:
                robot.disconnect()
            finally:
                camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
