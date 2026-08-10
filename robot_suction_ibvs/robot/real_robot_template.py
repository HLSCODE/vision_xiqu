"""Explicit TODO-only template for a real robot vendor SDK adapter."""

from __future__ import annotations

from typing import Sequence

from robot.base import RobotInterface


class RealRobot(RobotInterface):
    """Implement this adapter after choosing the robot's official SDK; do not edit control code."""

    def _todo(self, method: str) -> None:
        raise NotImplementedError(f"TODO: implement RealRobot.{method} with the verified robot SDK")

    def connect(self) -> None: self._todo("connect")
    def disconnect(self) -> None: self._todo("disconnect")
    def move_to_observe_pose(self) -> None: self._todo("move_to_observe_pose")
    def move_pose(self, pose: Sequence[float]) -> None: self._todo("move_pose")
    def move_xy_relative(self, dx_mm: float, dy_mm: float) -> None: self._todo("move_xy_relative")
    def move_z_absolute(self, z_mm: float, speed_mm_s: float) -> None: self._todo("move_z_absolute")
    def move_z_relative(self, dz_mm: float, speed_mm_s: float) -> None: self._todo("move_z_relative")
    def set_xy_velocity(self, vx_mm_s: float, vy_mm_s: float) -> None: self._todo("set_xy_velocity")
    def stop_xy(self) -> None: self._todo("stop_xy")
    def stop_all(self) -> None: self._todo("stop_all")
    def get_current_pose(self) -> Sequence[float]:
        self._todo("get_current_pose")
        return []
    def is_moving(self) -> bool:
        self._todo("is_moving")
        return False
    def wait_until_stop(self, timeout_s: float | None = None) -> None: self._todo("wait_until_stop")
