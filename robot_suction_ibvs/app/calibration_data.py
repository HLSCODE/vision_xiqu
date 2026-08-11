"""Calibration-array persistence with context metadata and integrity checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from app.config import AppConfig


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CalibrationMetadata:
    """Conditions under which one pixel-space calibration artifact is valid."""

    schema_version: int
    artifact_kind: str
    image_size: tuple[int, int]
    enable_undistort: bool
    intrinsic_sha256: str | None
    observe_pose: tuple[float, ...]
    work_frame_name: str
    work_frame_pose_m_rad: tuple[float, ...]
    servo_A_sha256: str | None = None


def metadata_path(data_path: Path) -> Path:
    """Return the JSON sidecar path for a NumPy calibration file."""
    return data_path.with_name(data_path.name + ".json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_intrinsic_sha256(config: AppConfig) -> str | None:
    """Fingerprint the active intrinsic file, or return None when correction is disabled."""
    if not config.camera.enable_undistort:
        return None
    path = config.path(config.camera.intrinsic_path)
    if not path.exists():
        raise FileNotFoundError(f"畸变校正已启用，但相机内参文件不存在：{path}")
    return file_sha256(path)


def make_metadata(
    config: AppConfig,
    artifact_kind: str,
    work_frame_name: str,
    work_frame_pose_m_rad: tuple[float, ...] | list[float],
    servo_A_sha256: str | None = None,
) -> CalibrationMetadata:
    frame_name = str(work_frame_name).strip("\x00 ")
    if not frame_name:
        raise ValueError("机械臂当前工作坐标系名称为空，不能保存可追溯标定")
    frame_pose = tuple(float(value) for value in work_frame_pose_m_rad)
    if len(frame_pose) != 6 or not all(math.isfinite(value) for value in frame_pose):
        raise ValueError("机械臂工作坐标系位姿必须包含六个有限值")
    return CalibrationMetadata(
        schema_version=SCHEMA_VERSION,
        artifact_kind=artifact_kind,
        image_size=(int(config.camera.width), int(config.camera.height)),
        enable_undistort=bool(config.camera.enable_undistort),
        intrinsic_sha256=current_intrinsic_sha256(config),
        observe_pose=tuple(float(value) for value in config.robot.observe_pose),
        work_frame_name=frame_name,
        work_frame_pose_m_rad=frame_pose,
        servo_A_sha256=servo_A_sha256,
    )


def save_calibration_array(path: Path, array: np.ndarray, metadata: CalibrationMetadata) -> None:
    """Save a NumPy array and its JSON sidecar using per-file atomic replacement."""
    values = np.asarray(array)
    if not np.all(np.isfinite(values)):
        raise ValueError("Calibration array contains NaN or infinity")
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = metadata_path(path)
    data_tmp = path.with_name(path.name + ".tmp")
    meta_tmp = sidecar.with_name(sidecar.name + ".tmp")
    with data_tmp.open("wb") as handle:
        np.save(handle, values)
    payload = asdict(metadata)
    payload["image_size"] = list(metadata.image_size)
    payload["observe_pose"] = list(metadata.observe_pose)
    payload["work_frame_pose_m_rad"] = list(metadata.work_frame_pose_m_rad)
    meta_tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    data_tmp.replace(path)
    meta_tmp.replace(sidecar)


def load_calibration_array(path: Path, expected_kind: str) -> tuple[np.ndarray, CalibrationMetadata]:
    """Load an artifact and reject missing, legacy, malformed, or mismatched metadata."""
    if not path.exists():
        raise FileNotFoundError(f"缺少标定文件：{path}")
    sidecar = metadata_path(path)
    if not sidecar.exists():
        raise FileNotFoundError(
            f"缺少标定元数据：{sidecar}。旧版仅含 .npy 的标定不能安全复用，请重新标定"
        )
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    try:
        if type(raw["enable_undistort"]) is not bool:
            raise TypeError("enable_undistort must be a JSON boolean")
        metadata = CalibrationMetadata(
            schema_version=int(raw["schema_version"]),
            artifact_kind=str(raw["artifact_kind"]),
            image_size=tuple(int(value) for value in raw["image_size"]),
            enable_undistort=bool(raw["enable_undistort"]),
            intrinsic_sha256=raw.get("intrinsic_sha256"),
            observe_pose=tuple(float(value) for value in raw["observe_pose"]),
            work_frame_name=str(raw["work_frame_name"]).strip("\x00 "),
            work_frame_pose_m_rad=tuple(float(value) for value in raw["work_frame_pose_m_rad"]),
            servo_A_sha256=raw.get("servo_A_sha256"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"标定元数据格式错误：{sidecar}") from exc
    if metadata.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"不支持的标定元数据版本 {metadata.schema_version}：{sidecar}，请重新标定"
        )
    if metadata.artifact_kind != expected_kind:
        raise ValueError(
            f"标定类型错误：期望 {expected_kind!r}，实际 {metadata.artifact_kind!r}"
        )
    if len(metadata.image_size) != 2 or any(value <= 0 for value in metadata.image_size):
        raise ValueError(f"标定图像尺寸无效：{metadata.image_size!r}")
    if len(metadata.observe_pose) != 6 or not all(math.isfinite(value) for value in metadata.observe_pose):
        raise ValueError("标定观察位姿必须包含六个有限数值")
    if not metadata.work_frame_name:
        raise ValueError("标定元数据缺少机械臂工作坐标系名称")
    if len(metadata.work_frame_pose_m_rad) != 6 or not all(
        math.isfinite(value) for value in metadata.work_frame_pose_m_rad
    ):
        raise ValueError("标定元数据中的工作坐标系位姿无效")
    for name, value in (
        ("intrinsic_sha256", metadata.intrinsic_sha256),
        ("servo_A_sha256", metadata.servo_A_sha256),
    ):
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
        ):
            raise ValueError(f"标定元数据中的 {name} 不是有效 SHA-256")
    array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"标定文件包含 NaN 或无穷大：{path}")
    return array, metadata


def validate_metadata_for_config(metadata: CalibrationMetadata, config: AppConfig) -> None:
    """Require runtime pixel geometry and observe pose to match calibration exactly."""
    expected_size = (int(config.camera.width), int(config.camera.height))
    if metadata.image_size != expected_size:
        raise ValueError(
            f"标定分辨率 {metadata.image_size} 与当前配置 {expected_size} 不一致，请恢复原分辨率或重新标定"
        )
    if metadata.enable_undistort != config.camera.enable_undistort:
        raise ValueError("标定时与当前运行时的畸变校正开关不一致，请恢复原设置或重新标定")
    if metadata.intrinsic_sha256 != current_intrinsic_sha256(config):
        raise ValueError("相机内参文件在标定后发生变化，请恢复原文件或重新执行像素标定")
    current_pose = np.asarray(config.robot.observe_pose, dtype=np.float64)
    calibrated_pose = np.asarray(metadata.observe_pose, dtype=np.float64)
    if not np.allclose(current_pose[:3], calibrated_pose[:3], atol=0.05, rtol=0.0) or not np.allclose(
        current_pose[3:], calibrated_pose[3:], atol=1e-4, rtol=0.0
    ):
        raise ValueError("robot.observe_pose 与标定时不一致，请恢复标定位姿或重新标定")


def require_matching_context(first: CalibrationMetadata, second: CalibrationMetadata) -> None:
    """Ensure servo and suction-reference files came from the same pixel geometry."""
    fields = (
        "image_size",
        "enable_undistort",
        "intrinsic_sha256",
        "observe_pose",
        "work_frame_name",
        "work_frame_pose_m_rad",
    )
    mismatches = [name for name in fields if getattr(first, name) != getattr(second, name)]
    if mismatches:
        raise ValueError("servo_A 与 suction_ref 标定上下文不一致：" + ", ".join(mismatches))
