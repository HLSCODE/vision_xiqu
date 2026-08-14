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


def move_to_baseline_xy(
    safety: SafetyManager,
    robot: RealRobot,
    baseline_pose: np.ndarray,
    tolerance_mm: float,
    max_corrections: int,
) -> np.ndarray:
    """Close the XY feedback loop until the original pose is reached or attempts run out.

    A relative return command is not assumed to be executed exactly. After each command
    the actual Cartesian feedback is read again and any remaining XY error is corrected.
    Z and orientation are never commanded by this calibration helper.
    """
    target_pose = np.asarray(baseline_pose, dtype=np.float64)
    if target_pose.shape != (6,) or not np.all(np.isfinite(target_pose)):
        raise ValueError("Calibration baseline pose must contain six finite values")

    for correction_count in range(max_corrections + 1):
        current_pose = read_robot_pose(robot, "Return-to-baseline feedback")
        feedback_error_xy = current_pose[:2] - target_pose[:2]
        feedback_error_mm = float(np.linalg.norm(feedback_error_xy))
        if feedback_error_mm <= tolerance_mm:
            print(
                "Returned robot XY feedback error="
                f"{feedback_error_mm:.4f}mm, dXY={feedback_error_xy.round(4).tolist()}mm, "
                f"corrections={correction_count}"
            )
            return current_pose

        if correction_count >= max_corrections:
            raise RuntimeError(
                "Robot XY feedback did not return to calibration baseline after "
                f"{max_corrections} corrections: {feedback_error_mm:.4f}mm > "
                f"{tolerance_mm:.4f}mm, dXY={feedback_error_xy.round(4).tolist()}mm"
            )

        correction_xy = -feedback_error_xy
        print(
            f"Return correction {correction_count + 1}/{max_corrections}: "
            f"error={feedback_error_mm:.4f}mm, command dXY={correction_xy.round(4).tolist()}mm"
        )
        safety.move_xy_relative(float(correction_xy[0]), float(correction_xy[1]))

    raise RuntimeError("Unreachable return-to-baseline state")


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
        help=(
            "Maximum image-return residual after subtracting the offset predicted "
            "from robot feedback (not the raw image displacement)"
        ),
    )
    parser.add_argument("--max-return-error-mm", type=float, default=0.10)
    parser.add_argument(
        "--max-return-corrections",
        type=int,
        default=3,
        help="Maximum feedback-based XY correction commands after each calibration jog",
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
        or args.max_return_error_mm <= 0
        or args.max_return_corrections < 1
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
        baseline_robot_pose = read_robot_pose(robot, "Calibration baseline")
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
        return_robot_offsets: list[np.ndarray] = []
        return_pixel_deltas: list[np.ndarray] = []
        print(
            "Move target setup complete. Sampling commanded XY increments at fixed observation height "
            f"in work frame {work_frame_name!r}."
        )
        for dx_mm, dy_mm in offsets_mm:
            # The previous return only needs to be within tolerance, so pair every
            # image baseline with the feedback pose from which this jog starts.
            sample_origin_pose = read_robot_pose(robot, "Pre-jog feedback")
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
                shifted_robot_pose = read_robot_pose(robot, "Post-jog feedback")
            finally:
                # 即使采样失败，也根据当前机械臂反馈修正回最初绝对 XY，而不是假定
                # 正反两条相对指令完全互逆。这样能暴露并补偿规划/反馈残差。
                returned_robot_pose = move_to_baseline_xy(
                    safety,
                    robot,
                    baseline_robot_pose,
                    args.max_return_error_mm,
                    args.max_return_corrections,
                )
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
            robot_offsets.append(actual_offset_xy)
            pixel_deltas.append(delta)
            return_robot_offsets.append(return_robot_offset_xy)
            return_pixel_deltas.append(return_pixel_delta)
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
        measured_return_offsets = np.stack(return_robot_offsets)
        measured_return_deltas = np.stack(return_pixel_deltas)
        predicted_return_deltas = measured_return_offsets @ matrix.T
        return_residual_vectors = measured_return_deltas - predicted_return_deltas
        return_residual_norms = np.linalg.norm(return_residual_vectors, axis=1)
        max_return_residual_px = float(np.max(return_residual_norms))
        return_residual_rms_px = float(np.sqrt(np.mean(return_residual_norms**2)))
        print("A (px/mm) =\n", matrix)
        print(
            f"fit_rms={fit_rms_px:.6f}px, rank={rank}, condition={condition:.3f}, "
            f"singular_values={singular_values}"
        )
        print(
            "feedback-explained return residual: "
            f"max={max_return_residual_px:.3f}px, rms={return_residual_rms_px:.3f}px"
        )
        if rank < 2:
            raise RuntimeError(f"Calibration fit rank is {rank}; both robot XY axes must be observable")
        if not np.isfinite(condition) or condition > 1e5:
            raise RuntimeError(f"Calibration matrix is singular or poorly conditioned: {condition:.3e}")
        if fit_rms_px > args.max_fit_rms_px:
            raise RuntimeError(
                f"Calibration fit RMS {fit_rms_px:.3f}px exceeds {args.max_fit_rms_px:.3f}px"
            )
        if max_return_residual_px > args.max_return_error_px:
            raise RuntimeError(
                "Image return is inconsistent with the corresponding robot feedback: "
                f"max unexplained residual {max_return_residual_px:.3f}px > "
                f"{args.max_return_error_px:.3f}px. The target/container/camera may have moved, "
                "or HSV detection may have switched contours."
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
