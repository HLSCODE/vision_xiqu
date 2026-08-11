"""通过两次图像采样标定吸管遮挡区外的预对准像素点。

前置条件：

1. ``calibrate_servo_xy.py`` 已生成 ``servo_A.npy``；
2. 对准位和预对准位使用相同相机分辨率、Z 高度与末端姿态；
3. 标定画面中只放置一个尺寸合格的静止目标。

操作员在 OpenCV 窗口中先于“吸管对准目标且目标可见”的状态按 A，再移动到
“吸管不遮挡目标”的预对准状态按 P。脚本直接测量两处目标像素中心之差作为
``prealign_offset_px``，同时把对准位目标中心保存为 ``suction_ref.npy``，再用
``servo_A`` 计算运行时最后一段相对 XY 位移。
"""

from __future__ import annotations

import argparse
import re
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
from vision.detector import HSVObjectDetector


def average_and_jitter(samples: deque[np.ndarray]) -> tuple[np.ndarray, float]:
    """返回最近连续检测中心的均值及相对均值的最大抖动，单位均为原图 px。"""
    if not samples:
        raise ValueError("No target centre samples")
    points = np.stack(samples).astype(np.float64)
    centre = np.mean(points, axis=0)
    jitter = float(np.max(np.linalg.norm(points - centre, axis=1)))
    return centre, jitter


def calculate_from_image_centres(
    servo_A_px_per_mm: np.ndarray,
    measured_aligned_px: np.ndarray,
    measured_prealign_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """根据两个实测目标中心计算预对准像素偏移和终末机械臂 XY。

    对准位中心会在完成全部安全检查后单独保存为 ``suction_ref.npy``。运行时终末
    位移满足 ``A @ final_xy = -prealign_offset_px``。
    """
    matrix = np.asarray(servo_A_px_per_mm, dtype=np.float64)
    aligned = np.asarray(measured_aligned_px, dtype=np.float64)
    prealign = np.asarray(measured_prealign_px, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("servo_A must be a finite 2x2 matrix")
    if aligned.shape != (2,) or prealign.shape != (2,):
        raise ValueError("All image centres must have shape (2,)")
    if not np.all(np.isfinite(aligned)) or not np.all(np.isfinite(prealign)):
        raise ValueError("Image centres contain NaN or infinity")
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > 1e5:
        raise ValueError(f"servo_A is singular or ill-conditioned (condition={condition:.2e})")

    prealign_offset_px = prealign - aligned
    final_xy_mm = np.linalg.inv(matrix) @ (-prealign_offset_px)
    return prealign_offset_px, final_xy_mm


def _format_number(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return "0.0" if text in {"0", "-0"} else text


def _replace_config_value(text: str, key: str, value: str) -> str:
    """替换单个现有 YAML 键，同时保留项目中的参数注释和排列顺序。"""
    pattern = re.compile(rf"^([ \t]*{re.escape(key)}[ \t]*:[ \t]*).*$", re.MULTILINE)
    updated, count = pattern.subn(rf"\g<1>{value}", text)
    if count != 1:
        raise ValueError(f"Expected exactly one '{key}' entry in config, found {count}")
    return updated


def apply_to_config(config_path: Path, offset_px: np.ndarray) -> None:
    """原子写入预对准像素偏移并打开人工确认开关。"""
    original = config_path.read_text(encoding="utf-8")
    offset_text = f"[{_format_number(offset_px[0])}, {_format_number(offset_px[1])}]"
    updated = _replace_config_value(original, "prealign_offset_px", offset_text)
    updated = _replace_config_value(updated, "prealign_offset_confirmed", "true")
    temporary = config_path.with_name(config_path.name + ".prealign.tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(config_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate visible prealignment from aligned and prealigned target image centres"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--frames", type=int, default=20, help="Recent valid centres averaged at each stage")
    parser.add_argument(
        "--max-jitter-px",
        type=float,
        default=3.0,
        help="Maximum sample distance from the averaged centre at each stationary stage",
    )
    parser.add_argument(
        "--apply-config",
        action="store_true",
        help="Write prealign_offset_px and set prealign_offset_confirmed=true",
    )
    parser.add_argument(
        "--confirm-target-visible",
        action="store_true",
        help="Required with --apply-config: confirms the target is fully visible at prealignment",
    )
    args = parser.parse_args()
    if args.frames < 3:
        parser.error("--frames must be at least 3")
    if args.max_jitter_px <= 0:
        parser.error("--max-jitter-px must be positive")
    if args.apply_config and not args.confirm_target_visible:
        parser.error("--apply-config requires --confirm-target-visible")

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    matrix_path = config.path(config.ibvs.servo_A_path)
    reference_path = config.path(config.suction.suction_ref_path)
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing required calibration file: {matrix_path}")
    matrix, servo_metadata = load_calibration_array(matrix_path, "servo_A")
    validate_metadata_for_config(servo_metadata, config)

    detector = HSVObjectDetector(
        config.vision,
        config.path(config.camera.intrinsic_path),
        config.camera.enable_undistort,
        expected_image_size=(config.camera.width, config.camera.height),
    )
    camera = OpenCVCamera(config.camera)
    samples: deque[np.ndarray] = deque(maxlen=args.frames)
    stage = "aligned"
    measured_aligned: np.ndarray | None = None

    print("Stage 1/2: keep the suction axis aligned above the target while the target is visible.")
    print("Wait for stable samples, then press A. Press Q/Esc at any time to cancel.")
    camera.open()
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                raise RuntimeError("Camera stream failed")
            result = detector.detect(frame)
            display = detector.preprocess(frame).copy()
            valid = detector.valid_objects(result.objects)
            current = valid[0] if len(valid) == 1 else None
            if current is None:
                samples.clear()
            else:
                samples.append(current.center.copy())
                point = tuple(np.round(current.center).astype(int))
                cv2.drawMarker(display, point, (0, 255, 255), cv2.MARKER_CROSS, 26, 2)

            if measured_aligned is not None:
                aligned_point = tuple(np.round(measured_aligned).astype(int))
                cv2.drawMarker(display, aligned_point, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 30, 2)

            jitter_text = "--"
            if samples:
                _, jitter = average_and_jitter(samples)
                jitter_text = f"{jitter:.2f}"
            instruction = "ALIGN: press A" if stage == "aligned" else "PREALIGN: press P"
            cv2.putText(display, instruction, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(
                display,
                f"valid targets={len(valid)} samples={len(samples)}/{args.frames} jitter={jitter_text}px",
                (12, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if current is not None else (0, 0, 255),
                2,
            )
            cv2.imshow("Prealignment calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 1

            expected_key = ord("a") if stage == "aligned" else ord("p")
            if key != expected_key:
                continue
            if len(samples) < args.frames:
                print(f"Need {args.frames - len(samples)} more consecutive valid frames")
                continue
            centre, jitter = average_and_jitter(samples)
            if jitter > args.max_jitter_px:
                print(
                    f"Target is not stable: jitter={jitter:.3f}px exceeds {args.max_jitter_px:.3f}px. "
                    "Keep the robot and target still, then try again."
                )
                samples.clear()
                continue

            if stage == "aligned":
                measured_aligned = centre
                stage = "prealign"
                samples.clear()
                print(f"Captured aligned target centre [u,v] = {centre.round(3).tolist()} px")
                print("Stage 2/2: move in XY until the same target is fully outside the suction occlusion.")
                print("Keep Z/orientation and the target fixed; wait for stable samples, then press P.")
                continue

            if measured_aligned is None:
                raise RuntimeError("Aligned target centre was not captured")
            measured_prealign = centre
            offset_px, final_xy_mm = calculate_from_image_centres(
                matrix,
                measured_aligned,
                measured_prealign,
            )
            offset_distance = float(np.linalg.norm(offset_px))
            final_distance = float(np.linalg.norm(final_xy_mm))
            print(f"measured_aligned_px        = {measured_aligned.round(6).tolist()}")
            print(f"measured_prealign_px       = {measured_prealign.round(6).tolist()}")
            print(f"new_suction_ref_px         = {measured_aligned.round(6).tolist()}")
            print(f"prealign_offset_px         = {offset_px.round(6).tolist()}")
            print(f"runtime_final_move_mm      = {final_xy_mm.round(6).tolist()}")
            print(f"final_move_distance_mm     = {final_distance:.6f}")

            if offset_distance <= config.ibvs.pixel_tolerance:
                raise ValueError("Prealignment offset is not larger than ibvs.pixel_tolerance")
            if final_distance > config.ibvs.final_approach_max_mm:
                raise ValueError(
                    f"Final move {final_distance:.3f} mm exceeds ibvs.final_approach_max_mm="
                    f"{config.ibvs.final_approach_max_mm:.3f} mm"
                )
            if final_distance > config.safety.max_xy_travel_mm:
                raise ValueError(
                    f"Final move {final_distance:.3f} mm exceeds safety.max_xy_travel_mm="
                    f"{config.safety.max_xy_travel_mm:.3f} mm"
                )

            print("Copy to config.yaml:")
            print(
                "  prealign_offset_px: "
                f"[{_format_number(offset_px[0])}, {_format_number(offset_px[1])}]"
            )
            print("  prealign_offset_confirmed: true")
            if args.apply_config:
                apply_to_config(config_path, offset_px)
                print(f"Updated {config_path}")
            # 两个位置均采样并通过抖动、矩阵和运动距离检查后，才覆盖吸管参考文件。
            reference_metadata = make_metadata(
                config,
                "suction_ref",
                servo_metadata.work_frame_name,
                servo_metadata.work_frame_pose_m_rad,
                servo_A_sha256=file_sha256(matrix_path),
            )
            save_calibration_array(reference_path, measured_aligned, reference_metadata)
            print(f"Saved suction_ref [u,v] px to {reference_path} and its context sidecar")
            return 0
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
