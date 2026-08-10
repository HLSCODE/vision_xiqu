"""Typed configuration loading for the complete system."""

from __future__ import annotations

from dataclasses import dataclass
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
    observe_pose: tuple[float, ...]
    observe_z_mm: float
    pick_z_mm: float
    safe_z_mm: float
    z_mode: str
    z_down_speed_mm_s: float
    z_up_speed_mm_s: float


@dataclass(frozen=True)
class SuctionConfig:
    hold_time_s: float
    suction_ref_path: str


@dataclass(frozen=True)
class SafetyConfig:
    max_xy_travel_mm: float
    camera_failure_limit: int


@dataclass(frozen=True)
class SystemConfig:
    show_debug: bool
    show_mask: bool
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

    return AppConfig(
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
