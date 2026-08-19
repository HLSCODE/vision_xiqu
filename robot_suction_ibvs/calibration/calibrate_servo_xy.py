"""Simple finite-difference calibration of the local 2D IBVS pixel/mm matrix.

For each of X-, X+, Y- and Y+, this script performs two independent trials:

1. Go to the configured fixed observation pose.
2. Wait for the camera and robot to settle, then record the target centre.
3. Move the requested XY increment and record the new centre.
4. Return to exactly the same fixed observation pose and record the centre again.

The return image is printed for the operator to inspect. It is deliberately not
used to alter the next trial or to compensate a movement. The matrix is fitted
from the eight measured robot XY displacements and their image-centre shifts.
"""

from __future__ import annotations

import argparse
from collections import deque
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
from vision.detector import RGBObjectDetector


def wait_for_settle(camera: OpenCVCamera, seconds: float, label: str) -> None:
    """Read and discard transition frames after a completed robot motion."""
    if seconds <= 0:
        return
    print(f"{label}: robot stopped; discarding transition frames for {seconds:.2f}s...")
    deadline = time.monotonic() + seconds
    discarded = 0
    while time.monotonic() < deadline:
        if camera.get_frame() is not None:
            discarded += 1
    print(f"{label}: discarded {discarded} transition frames; sampling stable centre")


def sample_stable_center(
    camera: OpenCVCamera,
    detector: RGBObjectDetector,
    frames: int,
    max_jitter_px: float,
    timeout_s: float,
    label: str,
) -> np.ndarray:
    """Return one target centre from a consecutive, stable image window.

    Any frame with zero or multiple eligible targets restarts the window. This
    prevents a centre from being averaged across different contours.
    """
    centres: deque[np.ndarray] = deque(maxlen=frames)
    deadline = time.monotonic() + timeout_s
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
        jitter = float(np.max(np.linalg.norm(points - centre, axis=1)))
        if jitter <= max_jitter_px:
            print(f"{label}: centre={centre.round(3).tolist()}px, jitter={jitter:.3f}px")
            return centre

    raise RuntimeError(
        f"{label}: no stable {frames}-frame centre within {timeout_s:.1f}s; "
        f"valid_frames={valid_frames}, invalid_frames={invalid_frames}. "
        "Keep one eligible target in view and check RGB detection."
    )


def read_robot_pose(robot: RealRobot, label: str) -> np.ndarray:
    """Read one valid Cartesian feedback pose in mm/rad."""
    pose = np.asarray(robot.get_current_pose(), dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise RuntimeError(f"{label}: robot returned an invalid Cartesian pose: {pose!r}")
    return pose


def fit_matrix(robot_offsets_mm: np.ndarray, pixel_offsets_px: np.ndarray) -> tuple[np.ndarray, int, float, float]:
    """Fit [du,dv]^T = A @ [dX,dY]^T from the eight finite-difference samples."""
    offsets = np.asarray(robot_offsets_mm, dtype=np.float64)
    pixels = np.asarray(pixel_offsets_px, dtype=np.float64)
    if offsets.shape != pixels.shape or offsets.ndim != 2 or offsets.shape[1] != 2:
        raise ValueError("Robot and pixel samples must both have shape (N, 2)")

    coefficients, _, rank, _ = np.linalg.lstsq(offsets, pixels, rcond=None)
    matrix = coefficients.T
    residuals = offsets @ coefficients - pixels
    rms_px = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))
    condition = float(np.linalg.cond(matrix))
    return matrix, int(rank), condition, rms_px


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple 8-trial XY pixel/mm calibration")
    parser.add_argument("--config", default="robot_suction_ibvs/config.yaml")
    parser.add_argument("--step-mm", type=float, default=2.0, help="Magnitude of every XY jog")
    parser.add_argument("--repeats", type=int, default=2, help="Trials for each of X-, X+, Y-, Y+")
    parser.add_argument("--frames", type=int, default=10, help="Stable frames averaged at each pose")
    parser.add_argument("--max-jitter-px", type=float, default=3.0)
    parser.add_argument("--settle-time-s", type=float, default=2.0)
    parser.add_argument("--sample-timeout-s", type=float, default=20.0)
    parser.add_argument("--max-fit-rms-px", type=float, default=15.0)
    parser.add_argument("--output", default=None, help="Defaults to ibvs.servo_A_path in config")
    args = parser.parse_args()

    if args.step_mm <= 0 or args.repeats < 1 or args.frames < 3:
        parser.error("--step-mm must be positive, --repeats must be at least 1, and --frames must be at least 3")
    if args.max_jitter_px <= 0 or args.settle_time_s < 0 or args.sample_timeout_s <= 0 or args.max_fit_rms_px <= 0:
        parser.error("jitter, settle-time, sample-timeout and fit-RMS limits are invalid")

    config = load_config(args.config)
    camera = OpenCVCamera(config.camera)
    robot = RealRobot(config.robot)
    detector = RGBObjectDetector(
        config.vision,
        config.path(config.camera.intrinsic_path),
        config.camera.enable_undistort,
        expected_image_size=(config.camera.width, config.camera.height),
    )
    safety = SafetyManager(robot, config.safety)
    output = config.path(args.output or config.ibvs.servo_A_path)

    directions = (
        ("X-", -args.step_mm, 0.0),
        ("X+", args.step_mm, 0.0),
        ("Y-", 0.0, -args.step_mm),
        ("Y+", 0.0, args.step_mm),
    )
    robot_offset_samples: list[np.ndarray] = []
    pixel_samples: list[np.ndarray] = []

    camera.open()
    try:
        robot.connect()
        work_frame_name = robot.get_current_work_frame_name()
        work_frame_pose = robot.get_current_work_frame_pose()
        print(
            "Fixed observation pose used before and after every trial: "
            f"{np.asarray(config.robot.observe_pose, dtype=np.float64).round(6).tolist()}"
        )
        print(f"Active work frame: {work_frame_name!r}")

        for direction, dx_mm, dy_mm in directions:
            for repeat_index in range(1, args.repeats + 1):
                label = f"{direction} trial {repeat_index}/{args.repeats}"

                # Every trial explicitly starts from the same configured observation pose.
                safety.move_to_observe_pose()
                wait_for_settle(camera, args.settle_time_s, f"{label} reference")
                reference = sample_stable_center(
                    camera,
                    detector,
                    args.frames,
                    args.max_jitter_px,
                    args.sample_timeout_s,
                    f"{label} reference",
                )
                reference_pose = read_robot_pose(robot, f"{label} reference feedback")

                try:
                    safety.move_xy_relative(dx_mm, dy_mm)
                    wait_for_settle(camera, args.settle_time_s, f"{label} moved")
                    moved = sample_stable_center(
                        camera,
                        detector,
                        args.frames,
                        args.max_jitter_px,
                        args.sample_timeout_s,
                        f"{label} moved",
                    )
                    moved_pose = read_robot_pose(robot, f"{label} moved feedback")
                finally:
                    # Always issue the configured absolute observation pose after a jog.
                    safety.move_to_observe_pose()

                wait_for_settle(camera, args.settle_time_s, f"{label} returned")
                returned = sample_stable_center(
                    camera,
                    detector,
                    args.frames,
                    args.max_jitter_px,
                    args.sample_timeout_s,
                    f"{label} returned",
                )

                pixel_delta = moved - reference
                feedback_delta = moved_pose[:2] - reference_pose[:2]
                if float(np.linalg.norm(feedback_delta)) <= 1e-6:
                    raise RuntimeError(f"{label}: robot feedback reported no XY displacement")
                return_delta = returned - reference
                robot_offset_samples.append(feedback_delta)
                pixel_samples.append(pixel_delta)
                print(
                    f"{label}: command dXY=[{dx_mm:+.3f}, {dy_mm:+.3f}]mm -> "
                    f"feedback dXY={feedback_delta.round(4).tolist()}mm -> "
                    f"dUV={pixel_delta.round(3).tolist()}px; "
                    f"return dUV={return_delta.round(3).tolist()}px"
                )

        robot_offsets = np.stack(robot_offset_samples)
        pixels = np.stack(pixel_samples)
        matrix, rank, condition, fit_rms_px = fit_matrix(robot_offsets, pixels)
        print("A (px/mm) =\n", matrix)
        print(f"rank={rank}, condition={condition:.3f}, fit_rms={fit_rms_px:.3f}px")

        if rank < 2:
            raise RuntimeError("Calibration rank is below 2; both X and Y must cause observable image motion")
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
