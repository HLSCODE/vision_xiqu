"""Fit the local 2x2 pixel/mm IBVS interaction matrix at observe height."""

from __future__ import annotations

import argparse
from collections import deque
import logging
import sys
import time
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
    max_wait_s: float,
    label: str,
) -> np.ndarray:
    """Wait for a stable sliding window containing exactly one eligible target."""
    centres: deque[np.ndarray] = deque(maxlen=frames)
    deadline = time.monotonic() + max_wait_s
    last_jitter: float | None = None
    valid_frames = 0
    invalid_frames = 0
    while time.monotonic() < deadline:
        frame = camera.get_frame()
        if frame is None:
            centres.clear()
            invalid_frames += 1
            continue
        objects = detector.valid_objects(detector.detect(frame).objects)
        if len(objects) != 1:
            centres.clear()
            invalid_frames += 1
            continue
        centres.append(objects[0].center.copy())
        valid_frames += 1
        if len(centres) < frames:
            continue
        points = np.stack(centres).astype(np.float64)
        centre = np.mean(points, axis=0)
        last_jitter = float(np.max(np.linalg.norm(points - centre, axis=1)))
        if last_jitter <= max_jitter_px:
            print(
                f"{label}: stable centre={centre.round(3).tolist()} px, "
                f"jitter={last_jitter:.3f}px"
            )
            return centre

    if len(centres) >= 2:
        points = np.stack(centres).astype(np.float64)
        u_range = (float(np.min(points[:, 0])), float(np.max(points[:, 0])))
        v_range = (float(np.min(points[:, 1])), float(np.max(points[:, 1])))
        range_text = (
            f"last_window_u=[{u_range[0]:.1f},{u_range[1]:.1f}]px, "
            f"last_window_v=[{v_range[0]:.1f},{v_range[1]:.1f}]px"
        )
    else:
        range_text = "last_window has fewer than two valid centres"
    jitter_text = "not available" if last_jitter is None else f"{last_jitter:.3f}px"
    raise RuntimeError(
        f"{label}: no stable {frames}-frame target window within {max_wait_s:.1f}s; "
        f"last jitter={jitter_text}, limit={max_jitter_px:.3f}px, "
        f"valid_frames={valid_frames}, invalid_frames={invalid_frames}, {range_text}. "
        "Keep exactly one eligible target still, wait for robot/camera exposure to settle, and check HSV contours."
    )


def wait_for_settle(camera: OpenCVCamera, seconds: float, label: str) -> int:
    """Discard every transition frame until the post-motion settling interval ends.

    ``SafetyManager.move_xy_relative`` has already waited for the robot trajectory to
    stop before this function is called. Frames received here may still contain
    structural vibration, rolling-shutter distortion or auto-exposure transients, so
    they are read and deliberately discarded. Target detection starts only afterwards.
    """
    if seconds <= 0:
        return 0
    print(f"{label}: robot stopped; discarding transition frames for {seconds:.2f}s...")
    deadline = time.monotonic() + seconds
    discarded_frames = 0
    while time.monotonic() < deadline:
        if camera.get_frame() is not None:
            discarded_frames += 1
    print(
        f"{label}: discarded {discarded_frames} transition frames; "
        "starting consecutive stable-centre detection"
    )
    return discarded_frames


def move_to_baseline_xy(
    safety: SafetyManager,
    robot: RealRobot,
    baseline_pose: np.ndarray,
    tolerance_mm: float,
) -> np.ndarray:
    """Use robot feedback to correct XY back to the original calibration pose."""
    current_pose = np.asarray(robot.get_current_pose(), dtype=np.float64)
    correction_xy = baseline_pose[:2] - current_pose[:2]
    if not np.all(np.isfinite(correction_xy)):
        raise RuntimeError("Robot returned non-finite XY feedback while restoring calibration baseline")
    if float(np.linalg.norm(correction_xy)) > 1e-6:
        safety.move_xy_relative(float(correction_xy[0]), float(correction_xy[1]))
    returned_pose = np.asarray(robot.get_current_pose(), dtype=np.float64)
    feedback_error_xy = returned_pose[:2] - baseline_pose[:2]
    feedback_error_mm = float(np.linalg.norm(feedback_error_xy))
    print(
        "Returned robot XY feedback error="
        f"{feedback_error_mm:.4f}mm, dXY={feedback_error_xy.round(4).tolist()}mm"
    )
    if feedback_error_mm > tolerance_mm:
        raise RuntimeError(
            f"Robot XY feedback did not return to calibration baseline: {feedback_error_mm:.4f}mm > "
            f"{tolerance_mm:.4f}mm"
        )
    return returned_pose


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate local 2D IBVS pixel/mm matrix A")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--frames", type=int, default=10, help="Frames averaged after every jog")
    parser.add_argument("--max-jitter-px", type=float, default=3.0)
    parser.add_argument("--sample-timeout-s", type=float, default=10.0)
    parser.add_argument("--settle-time-s", type=float, default=1.0)
    parser.add_argument("--max-return-error-px", type=float, default=3.0)
    parser.add_argument("--max-return-error-mm", type=float, default=0.10)
    parser.add_argument("--max-fit-rms-px", type=float, default=2.0)
    parser.add_argument("--output", default=None, help="Defaults to ibvs.servo_A_path in config")
    args = parser.parse_args()
    if args.frames < 3:
        parser.error("--frames must be at least 3")
    if (
        args.max_jitter_px <= 0
        or args.sample_timeout_s <= 0
        or args.settle_time_s < 0
        or args.max_return_error_px <= 0
        or args.max_return_error_mm <= 0
        or args.max_fit_rms_px <= 0
    ):
        parser.error("jitter, timeout, settle-time, return-error and fit-RMS limits are invalid")
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
    try:
        robot.connect()
        work_frame_name = robot.get_current_work_frame_name()
        work_frame_pose = robot.get_current_work_frame_pose()
        safety.move_to_observe_pose()
        safety.begin_alignment()
        baseline_robot_pose = np.asarray(robot.get_current_pose(), dtype=np.float64)
        wait_for_settle(camera, args.settle_time_s, "Observation pose")
        baseline = average_target_center(
            camera,
            detector,
            args.frames,
            args.max_jitter_px,
            args.sample_timeout_s,
            "Baseline",
        )
        pixel_deltas: list[np.ndarray] = []
        print(
            "Move target setup complete. Sampling commanded XY increments at fixed observation height "
            f"in work frame {work_frame_name!r}."
        )
        for dx_mm, dy_mm in offsets_mm:
            safety.move_xy_relative(float(dx_mm), float(dy_mm))
            try:
                wait_for_settle(
                    camera,
                    args.settle_time_s,
                    f"Shift ({dx_mm:+.1f},{dy_mm:+.1f})mm",
                )
                shifted = average_target_center(
                    camera,
                    detector,
                    args.frames,
                    args.max_jitter_px,
                    args.sample_timeout_s,
                    f"Shift ({dx_mm:+.1f},{dy_mm:+.1f})mm",
                )
            finally:
                # 即使采样失败，也根据当前机械臂反馈修正回最初绝对 XY，而不是假定
                # 正反两条相对指令完全互逆。这样能暴露并补偿规划/反馈残差。
                move_to_baseline_xy(
                    safety,
                    robot,
                    baseline_robot_pose,
                    args.max_return_error_mm,
                )
            delta = shifted - baseline
            wait_for_settle(camera, args.settle_time_s, "Returned baseline")
            returned = average_target_center(
                camera,
                detector,
                args.frames,
                args.max_jitter_px,
                args.sample_timeout_s,
                "Returned baseline",
            )
            return_error = float(np.linalg.norm(returned - baseline))
            if return_error > args.max_return_error_px:
                raise RuntimeError(
                    f"Image target did not return to baseline after robot XY feedback correction: "
                    f"{return_error:.3f}px > {args.max_return_error_px:.3f}px. "
                    "If robot feedback error above was small, the target/container/camera moved or the detector "
                    "locked onto a different contour; inspect detection before retrying."
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
        cleanup_failures: list[str] = []
        if robot.connected:
            try:
                safety.stop_all()
            except Exception as exc:
                cleanup_failures.append(f"stop robot: {exc}")
        try:
            robot.disconnect()
        except Exception as exc:
            cleanup_failures.append(f"disconnect robot: {exc}")
        try:
            camera.close()
        except Exception as exc:
            cleanup_failures.append(f"close camera: {exc}")
        if cleanup_failures:
            print("Cleanup warnings: " + "; ".join(cleanup_failures), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
