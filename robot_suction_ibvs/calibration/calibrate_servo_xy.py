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


def read_robot_pose(robot: RealRobot, label: str) -> np.ndarray:
    """Read and validate one six-axis Cartesian pose in mm/rad."""
    pose = np.asarray(robot.get_current_pose(), dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise RuntimeError(f"{label}: robot returned an invalid Cartesian pose: {pose!r}")
    return pose


def fit_servo_matrix(
    robot_offsets_mm: np.ndarray,
    pixel_deltas_px: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray, float]:
    """Fit dUV = A @ dXY from measured robot feedback displacements."""
    offsets = np.asarray(robot_offsets_mm, dtype=np.float64)
    deltas = np.asarray(pixel_deltas_px, dtype=np.float64)
    if offsets.ndim != 2 or offsets.shape[1] != 2 or deltas.shape != offsets.shape:
        raise ValueError("Robot offsets and pixel deltas must both have shape (N, 2)")
    if not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(deltas)):
        raise ValueError("Calibration samples contain NaN or infinity")

    coefficients, _, rank, singular_values = np.linalg.lstsq(offsets, deltas, rcond=None)
    matrix = coefficients.T
    residual_vectors = offsets @ coefficients - deltas
    fit_rms_px = float(np.sqrt(np.mean(np.sum(residual_vectors * residual_vectors, axis=1))))
    return matrix, int(rank), singular_values, fit_rms_px


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate local 2D IBVS pixel/mm matrix A")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--frames", type=int, default=10, help="Frames averaged after every jog")
    parser.add_argument("--max-jitter-px", type=float, default=3.0)
    parser.add_argument("--sample-timeout-s", type=float, default=10.0)
    parser.add_argument("--settle-time-s", type=float, default=1.0)
    parser.add_argument(
        "--max-return-error-px",
        type=float,
        default=3.0,
        help="Maximum raw image-centre difference after returning to observe_pose",
    )
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
        wait_for_settle(camera, args.settle_time_s, "Observation pose")
        baseline = average_target_center(
            camera,
            detector,
            args.frames,
            args.max_jitter_px,
            args.sample_timeout_s,
            "Baseline",
        )
        robot_offsets: list[np.ndarray] = []
        pixel_deltas: list[np.ndarray] = []
        print(
            "Confirmed absolute observe_pose used for every return: "
            f"{np.asarray(config.robot.observe_pose, dtype=np.float64).round(6).tolist()}"
        )
        print(
            "Move target setup complete. Sampling commanded XY increments at fixed observation height "
            f"in work frame {work_frame_name!r}."
        )
        for dx_mm, dy_mm in offsets_mm:
            # Pair every image baseline with the feedback pose from which this jog starts.
            sample_origin_pose = read_robot_pose(robot, "Pre-jog feedback")
            try:
                safety.move_xy_relative(float(dx_mm), float(dy_mm))
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
                shifted_robot_pose = read_robot_pose(robot, "Post-jog feedback")
            finally:
                # 用与标定起始时相同的已确认观察位绝对回位。这与手动输入原位姿
                # 的流程一致，不能由中间反馈位姿再拼出一个相对回位目标。
                safety.move_to_observe_pose()
                returned_robot_pose = read_robot_pose(robot, "Returned observation-pose feedback")
            actual_offset_xy = shifted_robot_pose[:2] - sample_origin_pose[:2]
            if float(np.linalg.norm(actual_offset_xy)) <= 1e-6:
                raise RuntimeError(
                    f"Robot feedback reported no XY motion for command ({dx_mm:+.1f},{dy_mm:+.1f})mm"
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
            return_pixel_delta = returned - baseline
            return_robot_offset_xy = returned_robot_pose[:2] - sample_origin_pose[:2]
            raw_return_error = float(np.linalg.norm(return_pixel_delta))
            if raw_return_error > args.max_return_error_px:
                raise RuntimeError(
                    "Image target did not return to the confirmed observation pose: "
                    f"{raw_return_error:.3f}px > {args.max_return_error_px:.3f}px, "
                    f"dUV={return_pixel_delta.round(3).tolist()}px. "
                    "Re-check config.robot.observe_pose, the active work frame, and target/detector stability."
                )
            robot_offsets.append(actual_offset_xy)
            pixel_deltas.append(delta)
            baseline = returned
            print(
                f"command dXY=({dx_mm:+.1f}, {dy_mm:+.1f})mm, "
                f"feedback dXY={actual_offset_xy.round(4).tolist()}mm -> "
                f"dUV={delta.round(3).tolist()}px, "
                f"raw_return_shift={raw_return_error:.3f}px, "
                f"return_feedback_dXY={return_robot_offset_xy.round(4).tolist()}mm"
            )
        measured_offsets = np.stack(robot_offsets)
        measured_deltas = np.stack(pixel_deltas)
        matrix, rank, singular_values, fit_rms_px = fit_servo_matrix(
            measured_offsets,
            measured_deltas,
        )
        condition = float(np.linalg.cond(matrix))
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
