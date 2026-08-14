"""Typed configuration loading for the complete system."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    fps: int
    device_id: int | str
    video_path: str | None
    enable_undistort: bool
    intrinsic_path: str


@dataclass(frozen=True)
class VisionConfig:
    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    min_area_px: float
    morphology_kernel: int
    size_threshold_px: float


@dataclass(frozen=True)
class TrackingConfig:
    max_distance_px: float
    max_lost_frames: int
    empty_scene_confirm_frames: int


@dataclass(frozen=True)
class IBVSConfig:
    gain: float
    max_velocity_mm_s: float
    min_velocity_mm_s: float
    pixel_tolerance: float
    stable_frames: int
    max_align_time_s: float
    error_growth_frames: int
    slowdown_error_px: float
    max_acceleration_mm_s2: float
    servo_A_path: str


@dataclass(frozen=True)
class RobotConfig:
    ip: str
    port: int
    api_timeout_ms: int
    observe_pose: tuple[float, ...]
    observe_pose_confirmed: bool
    observe_speed_percent: int
    linear_speed_percent: int
    velocity_period_ms: int
    motion_timeout_s: float
    observe_z_mm: float
    pick_z_mm: float
    safe_z_mm: float
    z_motion_confirmed: bool
    z_mode: str
    z_down_speed_mm_s: float
    z_up_speed_mm_s: float


@dataclass(frozen=True)
class SuctionConfig:
    hold_time_s: float
    suction_ref_path: str
    serial_port: str
    baudrate: int
    timeout_s: float
    max_retries: int
    retry_delay_s: float
    response_bytes: int
    require_response: bool
    initialize_before_first_absorb: bool
    absorb_volume_ul: int
    absorb_speed_ul_s: int | None


@dataclass(frozen=True)
class SafetyConfig:
    max_xy_travel_mm: float
    camera_failure_limit: int


@dataclass(frozen=True)
class SystemConfig:
    save_csv: bool
    loop_hz: float


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    camera: CameraConfig
    vision: VisionConfig
    tracking: TrackingConfig
    ibvs: IBVSConfig
    robot: RobotConfig
    suction: SuctionConfig
    safety: SafetyConfig
    system: SystemConfig

    def path(self, relative_path: str) -> Path:
        """Resolve a configured project-relative path."""
        path = Path(relative_path)
        return path if path.is_absolute() else self.root_dir / path


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in raw or not isinstance(raw[key], dict):
        raise ValueError(f"Missing or invalid '{key}' section in config.yaml")
    return raw[key]


def _require_positive(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate(config: AppConfig) -> None:
    """Reject unsafe or internally inconsistent settings before hardware startup."""
    _require_positive_int("camera.width", config.camera.width)
    _require_positive_int("camera.height", config.camera.height)
    _require_positive_int("camera.fps", config.camera.fps)

    for name, values, limits in (
        ("vision.hsv_lower", config.vision.hsv_lower, (179, 255, 255)),
        ("vision.hsv_upper", config.vision.hsv_upper, (179, 255, 255)),
    ):
        if len(values) != 3 or any(value < 0 or value > limit for value, limit in zip(values, limits)):
            raise ValueError(f"{name} must be [H,S,V] within OpenCV HSV ranges")
    if any(lower > upper for lower, upper in zip(config.vision.hsv_lower, config.vision.hsv_upper)):
        raise ValueError("vision.hsv_lower must not exceed vision.hsv_upper")
    _require_positive("vision.min_area_px", config.vision.min_area_px)
    _require_positive("vision.morphology_kernel", config.vision.morphology_kernel)
    _require_positive("vision.size_threshold_px", config.vision.size_threshold_px)

    _require_positive("tracking.max_distance_px", config.tracking.max_distance_px)
    _require_positive_int("tracking.max_lost_frames", config.tracking.max_lost_frames)
    _require_positive_int("tracking.empty_scene_confirm_frames", config.tracking.empty_scene_confirm_frames)

    _require_positive("ibvs.gain", config.ibvs.gain)
    _require_positive("ibvs.max_velocity_mm_s", config.ibvs.max_velocity_mm_s)
    if not math.isfinite(config.ibvs.min_velocity_mm_s) or config.ibvs.min_velocity_mm_s < 0:
        raise ValueError("ibvs.min_velocity_mm_s must be non-negative and finite")
    if config.ibvs.min_velocity_mm_s > config.ibvs.max_velocity_mm_s:
        raise ValueError("ibvs.min_velocity_mm_s must not exceed ibvs.max_velocity_mm_s")
    _require_positive("ibvs.pixel_tolerance", config.ibvs.pixel_tolerance)
    _require_positive_int("ibvs.stable_frames", config.ibvs.stable_frames)
    _require_positive("ibvs.max_align_time_s", config.ibvs.max_align_time_s)
    _require_positive_int("ibvs.error_growth_frames", config.ibvs.error_growth_frames)
    if not math.isfinite(config.ibvs.slowdown_error_px) or config.ibvs.slowdown_error_px < 0:
        raise ValueError("ibvs.slowdown_error_px must be non-negative and finite")
    _require_positive("ibvs.max_acceleration_mm_s2", config.ibvs.max_acceleration_mm_s2)
    if len(config.robot.observe_pose) != 6 or not all(math.isfinite(value) for value in config.robot.observe_pose):
        raise ValueError("robot.observe_pose must contain six finite values")
    if not str(config.robot.ip).strip():
        raise ValueError("robot.ip must not be empty")
    _require_positive_int("robot.port", config.robot.port)
    _require_positive_int("robot.api_timeout_ms", config.robot.api_timeout_ms)
    for name, value in (
        ("robot.observe_speed_percent", config.robot.observe_speed_percent),
        ("robot.linear_speed_percent", config.robot.linear_speed_percent),
    ):
        if not 1 <= int(value) <= 100:
            raise ValueError(f"{name} must be within 1..100")
    _require_positive_int("robot.velocity_period_ms", config.robot.velocity_period_ms)
    _require_positive("robot.motion_timeout_s", config.robot.motion_timeout_s)
    for name, value in (
        ("robot.observe_z_mm", config.robot.observe_z_mm),
        ("robot.pick_z_mm", config.robot.pick_z_mm),
        ("robot.safe_z_mm", config.robot.safe_z_mm),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    _require_positive("robot.z_down_speed_mm_s", config.robot.z_down_speed_mm_s)
    _require_positive("robot.z_up_speed_mm_s", config.robot.z_up_speed_mm_s)
    if config.robot.z_mode not in {"absolute", "relative"}:
        raise ValueError("robot.z_mode must be 'absolute' or 'relative'")

    if not math.isfinite(config.suction.hold_time_s) or config.suction.hold_time_s < 0:
        raise ValueError("suction.hold_time_s must be non-negative and finite")
    if not str(config.suction.serial_port).strip():
        raise ValueError("suction.serial_port must not be empty")
    _require_positive_int("suction.baudrate", config.suction.baudrate)
    _require_positive("suction.timeout_s", config.suction.timeout_s)
    _require_positive_int("suction.max_retries", config.suction.max_retries)
    if not math.isfinite(config.suction.retry_delay_s) or config.suction.retry_delay_s < 0:
        raise ValueError("suction.retry_delay_s must be non-negative and finite")
    _require_positive_int("suction.response_bytes", config.suction.response_bytes)
    if not 1 <= config.suction.absorb_volume_ul <= 0xFFFF:
        raise ValueError("suction.absorb_volume_ul must be within 1..65535 uL")
    if config.suction.absorb_speed_ul_s is not None and not 1 <= config.suction.absorb_speed_ul_s <= 9999:
        raise ValueError("suction.absorb_speed_ul_s must be null or within 1..9999 uL/s")
    if config.suction.absorb_speed_ul_s is not None:
        minimum_hold_s = config.suction.absorb_volume_ul / config.suction.absorb_speed_ul_s
        if config.suction.hold_time_s < minimum_hold_s:
            raise ValueError(
                "suction.hold_time_s must be at least absorb_volume_ul / absorb_speed_ul_s "
                f"({minimum_hold_s:.3f}s)"
            )
    _require_positive("safety.max_xy_travel_mm", config.safety.max_xy_travel_mm)
    _require_positive_int("safety.camera_failure_limit", config.safety.camera_failure_limit)
    _require_positive("system.loop_hz", config.system.loop_hz)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate config.yaml into immutable, typed settings."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("config.yaml must contain a YAML mapping")

    camera = _section(raw, "camera")
    vision = _section(raw, "vision")
    tracking = _section(raw, "tracking")
    ibvs = _section(raw, "ibvs")
    robot = _section(raw, "robot")
    suction = _section(raw, "suction")
    safety = _section(raw, "safety")
    system = _section(raw, "system")

    config = AppConfig(
        root_dir=config_path.parent,
        camera=CameraConfig(**camera),
        vision=VisionConfig(
            **{**vision, "hsv_lower": tuple(vision["hsv_lower"]), "hsv_upper": tuple(vision["hsv_upper"])}
        ),
        tracking=TrackingConfig(**tracking),
        ibvs=IBVSConfig(**ibvs),
        robot=RobotConfig(**{**robot, "observe_pose": tuple(robot["observe_pose"])}),
        suction=SuctionConfig(**suction),
        safety=SafetyConfig(**safety),
        system=SystemConfig(**system),
    )
    _validate(config)
    return config
