"""Vendor-neutral robot interface required by the controller."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class RobotInterface(ABC):
    """All robot operations use mm, mm/s, and a vendor-defined pose sequence."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def move_to_observe_pose(self) -> None: ...

    @abstractmethod
    def move_pose(self, pose: Sequence[float]) -> None: ...

    @abstractmethod
    def move_xy_relative(self, dx_mm: float, dy_mm: float) -> None: ...

    @abstractmethod
    def move_z_absolute(self, z_mm: float, speed_mm_s: float) -> None: ...

    @abstractmethod
    def move_z_relative(self, dz_mm: float, speed_mm_s: float) -> None: ...

    @abstractmethod
    def set_xy_velocity(self, vx_mm_s: float, vy_mm_s: float) -> None: ...

    @abstractmethod
    def stop_xy(self) -> None: ...

    @abstractmethod
    def stop_all(self) -> None: ...

    @abstractmethod
    def get_current_pose(self) -> Sequence[float]: ...

    @abstractmethod
    def get_current_work_frame_name(self) -> str: ...

    @abstractmethod
    def get_current_work_frame_pose(self) -> Sequence[float]: ...

    @abstractmethod
    def is_moving(self) -> bool: ...

    @abstractmethod
    def wait_until_stop(self, timeout_s: float | None = None) -> None: ...
