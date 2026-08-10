"""Locally calibrated two-axis image-based visual servo controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import IBVSConfig


@dataclass(slots=True)
class IBVSCommand:
    """One 2D visual-servo command with explicit pixel and mm/s units."""

    velocity_mm_s: np.ndarray  # shape (2,), [vx, vy], mm/s
    error_px: np.ndarray  # shape (2,), [eu, ev], px
    error_norm_px: float


class IBVSController:
    """Compute `v_xy = -gain * inv(A) * pixel_error` with velocity safeguards."""

    def __init__(self, servo_A_px_per_mm: np.ndarray, config: IBVSConfig) -> None:
        matrix = np.asarray(servo_A_px_per_mm, dtype=np.float64)
        if matrix.shape != (2, 2):
            raise ValueError(f"servo_A must have shape (2, 2), got {matrix.shape}")
        # 条件数过大时求逆会放大像素噪声，可能导致机械臂出现不可控速度。
        condition = float(np.linalg.cond(matrix))
        if not np.isfinite(condition) or condition > 1e5:
            raise ValueError(f"servo_A is singular or ill-conditioned (condition={condition:.2e})")
        self.A_px_per_mm = matrix
        self.A_inverse_mm_per_px = np.linalg.inv(matrix)
        self.condition_number = condition
        self.config = config
        self._last_velocity_mm_s = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        """Clear velocity history before a new alignment task."""
        self._last_velocity_mm_s[:] = 0.0

    def compute(self, current_pixel_px: np.ndarray, reference_pixel_px: np.ndarray, dt_s: float) -> IBVSCommand:
        """Calculate a slew-rate-limited XY command from current image centre."""
        error = np.asarray(current_pixel_px, dtype=np.float64) - np.asarray(reference_pixel_px, dtype=np.float64)
        error_norm = float(np.linalg.norm(error))
        # A 的单位为 px/mm，因此 A^{-1}@error 的单位为 mm；负号使误差朝零收敛。
        velocity = -self.config.gain * (self.A_inverse_mm_per_px @ error)

        # 接近参考点时按误差缩放速度，减小过冲和高速突停的风险。
        if self.config.slowdown_error_px > 0:
            scale = min(1.0, error_norm / self.config.slowdown_error_px)
            velocity *= scale
        speed = float(np.linalg.norm(velocity))
        if speed > self.config.max_velocity_mm_s:
            velocity *= self.config.max_velocity_mm_s / speed
        if self.config.min_velocity_mm_s > 0 and 0 < speed < self.config.min_velocity_mm_s:
            velocity *= self.config.min_velocity_mm_s / speed
        if error_norm <= self.config.pixel_tolerance:
            # 对准判定由状态机连续多帧完成；这里先保证单帧进入容差就不再产生 XY 速度。
            velocity[:] = 0.0

        # 相邻控制周期的速度变化受加速度上限约束，形成简单的 slew-rate limiter。
        max_delta = self.config.max_acceleration_mm_s2 * max(dt_s, 1e-3)
        delta = velocity - self._last_velocity_mm_s
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > max_delta:
            velocity = self._last_velocity_mm_s + delta * (max_delta / delta_norm)
        self._last_velocity_mm_s = velocity.copy()
        return IBVSCommand(velocity_mm_s=velocity, error_px=error, error_norm_px=error_norm)
