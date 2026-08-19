"""睿尔曼机械臂 Python API2 真实硬件适配器。"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Sequence

from app.config import RobotConfig
from robot.base import RobotInterface


_SDK_STATUS = {
    1: "控制器拒绝指令、参数错误或机械臂状态异常",
    -1: "数据发送失败或连接句柄无效",
    -2: "数据接收失败或控制器响应超时",
    -3: "控制器返回数据解析失败",
    -4: "当前到位设备校验失败或控制器不支持该接口",
    -5: "运动指令等待超时",
    -6: "运动被外部停止指令终止",
    -7: "当前控制器版本不支持该接口",
}


class RealManSDKError(RuntimeError):
    """睿尔曼 SDK 返回非零状态码或连接状态无效。"""


class RealRobot(RobotInterface):
    """将工程的 mm/mm·s⁻¹ 接口映射到睿尔曼 API2。

    工程位姿格式固定为 ``[x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]``；
    睿尔曼 SDK 的笛卡尔位置使用米，因此所有位置都在本类边界完成换算。
    XY 视觉伺服使用 ``rm_movev_canfd``，速度参考系为当前工作坐标系。
    """

    # API2 的轨迹状态在规划刚下发时可能短暂仍为“未运动”。因此不能只看
    # trajectory_type；必须同时确认末端反馈已到达本次规划的目标位姿。
    _POSITION_REACHED_TOLERANCE_MM = 0.5
    _ORIENTATION_REACHED_TOLERANCE_RAD = 0.02

    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        self._arm: Any | None = None
        self._connection_lock = threading.RLock()
        self._max_line_speed_m_s: float | None = None
        self._robot_info: dict[str, Any] | None = None
        self._last_planned_motion_at: float | None = None
        self._last_planned_target_pose_mm: tuple[float, ...] | None = None
        self._motion_commanded = False

    @property
    def connected(self) -> bool:
        with self._connection_lock:
            return self._arm is not None

    @property
    def robot_info(self) -> dict[str, Any] | None:
        return None if self._robot_info is None else dict(self._robot_info)

    def connect(self) -> None:
        """连接控制器、校验上电状态并初始化笛卡尔速度透传。"""
        with self._connection_lock:
            if self._arm is not None:
                return

        try:
            # 延迟导入：未安装机械臂 SDK 时，相机预览界面仍可正常启动。
            from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
        except (ImportError, OSError) as exc:
            raise RealManSDKError(
                "无法加载睿尔曼 Robotic_Arm SDK。请在当前 Python 环境安装 API2，"
                "并确认其 libs 目录包含与系统架构匹配的动态库。"
            ) from exc

        arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle_created = False
        try:
            arm.rm_set_timeout(int(self.config.api_timeout_ms))
            handle = arm.rm_create_robot_arm(
                self.config.ip,
                int(self.config.port),
                level=2,
            )
            if int(handle.id) == -1:
                raise RealManSDKError(
                    f"连接睿尔曼机械臂失败：{self.config.ip}:{self.config.port} 返回无效句柄"
                )
            handle_created = True

            code, info = arm.rm_get_robot_info()
            self._check_code("读取机械臂型号信息", code)

            code, power_state = arm.rm_get_arm_power_state()
            self._check_code("读取机械臂上电状态", code)
            if int(power_state) != 1:
                raise RealManSDKError(
                    "机械臂控制器已连接，但机械臂未上电；请在示教器确认上电和急停状态"
                )

            code, state = arm.rm_get_current_arm_state()
            self._check_code("读取机械臂当前状态", code)
            errors = state.get("err", {}) if isinstance(state, dict) else {}
            if code != 0:
                raise RealManSDKError(
                    f"机械臂存在未处理错误 {errors.get('err', [])}；请先在示教器排除，程序不会自动清错"
                )

            code, max_line_speed = arm.rm_get_arm_max_line_speed()
            self._check_code("读取末端最大线速度", code)
            if not math.isfinite(max_line_speed) or max_line_speed <= 0:
                raise RealManSDKError(f"机械臂返回无效最大线速度：{max_line_speed!r}")

            # 1 表示当前工作坐标系；IBVS 标定和在线速度必须使用同一坐标系。
            code = arm.rm_set_movev_canfd_init(
                1,
                1,
                int(self.config.velocity_period_ms),
            )
            self._check_code("初始化笛卡尔速度透传", code)
        except Exception:
            # 官方 API 将“删除单个句柄”和“销毁全部全局线程”作为两种独立释放方式。
            # 已创建句柄时只删除该句柄，不能紧接着再次调用全局 rm_destroy()。
            if handle_created:
                try:
                    arm.rm_delete_robot_arm()
                except Exception:
                    pass
            else:
                try:
                    type(arm).rm_destroy()
                except Exception:
                    pass
            raise

        with self._connection_lock:
            self._arm = arm
            self._max_line_speed_m_s = float(max_line_speed)
            self._robot_info = dict(info)

    def disconnect(self) -> None:
        """先停止当前轨迹，再按官方单连接流程删除机械臂句柄。"""
        with self._connection_lock:
            arm = self._arm
            self._arm = None
            self._max_line_speed_m_s = None
            self._robot_info = None
            self._last_planned_motion_at = None
            self._last_planned_target_pose_mm = None
            motion_commanded = self._motion_commanded
            self._motion_commanded = False
        if arm is None:
            return

        failures: list[str] = []
        if motion_commanded:
            try:
                code = arm.rm_set_arm_stop()
                if code != 0:
                    failures.append(self._format_error("断开前停止机械臂", code))
            except Exception as exc:
                failures.append(f"断开前停止机械臂异常：{exc}")
        try:
            code = arm.rm_delete_robot_arm()
            if code != 0:
                failures.append(self._format_error("删除机械臂连接句柄", code))
        except Exception as exc:
            failures.append(f"删除机械臂连接句柄异常：{exc}")
        # 不在 rm_delete_robot_arm() 后继续调用全局 rm_destroy()。API2 1.1.3 的
        # 全局销毁会再次处理连接/线程；连续使用两套释放路径可能导致原生库崩溃，
        # 也会破坏 GUI 后续重新连接同一机械臂的能力。
        if failures:
            raise RealManSDKError("；".join(failures))

    def move_to_observe_pose(self) -> None:
        """以关节规划方式移动到配置的固定观察位。"""
        if not self.config.observe_pose_confirmed:
            raise RealManSDKError(
                "观察位尚未确认：请在 config.yaml 填写实际示教的 observe_pose，"
                "核对安全后把 observe_pose_confirmed 改为 true"
            )
        self._move_joint_to_pose(
            self.config.observe_pose,
            int(self.config.observe_speed_percent),
            "移动到观察位",
        )

    def move_pose(self, pose: Sequence[float]) -> None:
        """以关节规划方式移动到工程坐标位姿。"""
        self._move_joint_to_pose(
            pose,
            int(self.config.observe_speed_percent),
            "移动到目标位姿",
        )

    def move_pose_linear(self, pose: Sequence[float]) -> None:
        """以笛卡尔直线规划移动到工程坐标绝对位姿。"""
        self._move_linear(
            pose,
            int(self.config.linear_speed_percent),
            "直线移动到目标位姿",
        )

    def move_xy_relative(self, dx_mm: float, dy_mm: float) -> None:
        """保持 Z 和姿态不变，在当前工作坐标系执行相对 XY 直线运动。"""
        self._validate_finite(dx_mm, dy_mm)
        pose = list(self.get_current_pose())
        pose[0] += float(dx_mm)
        pose[1] += float(dy_mm)
        self._move_linear(
            pose,
            int(self.config.linear_speed_percent),
            "相对 XY 直线运动",
        )

    def move_z_absolute(self, z_mm: float, speed_mm_s: float) -> None:
        """保持 XY 和姿态不变，直线移动到工作坐标系绝对 Z。"""
        self._require_z_motion_confirmed()
        self._validate_finite(z_mm, speed_mm_s)
        pose = list(self.get_current_pose())
        pose[2] = float(z_mm)
        self._move_linear(pose, self._speed_percent(speed_mm_s), "绝对 Z 直线运动")

    def move_z_relative(self, dz_mm: float, speed_mm_s: float) -> None:
        """保持 XY 和姿态不变，执行工作坐标系相对 Z 直线运动。"""
        self._require_z_motion_confirmed()
        self._validate_finite(dz_mm, speed_mm_s)
        pose = list(self.get_current_pose())
        pose[2] += float(dz_mm)
        self._move_linear(pose, self._speed_percent(speed_mm_s), "相对 Z 直线运动")

    def set_xy_velocity(self, vx_mm_s: float, vy_mm_s: float) -> None:
        """下发工作坐标系 XY 速度；SDK 平移速度单位为 m/s。"""
        self._validate_finite(vx_mm_s, vy_mm_s)
        arm = self._require_connected()
        max_speed = self._max_line_speed_m_s
        speed_m_s = math.hypot(float(vx_mm_s), float(vy_mm_s)) / 1000.0
        if max_speed is None or speed_m_s > max_speed:
            raise ValueError(
                f"XY 速度 {speed_m_s:.6f} m/s 超过机械臂最大线速度 {max_speed!r} m/s"
            )
        velocity_m_s = [
            float(vx_mm_s) / 1000.0,
            float(vy_mm_s) / 1000.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        # 30 Hz 控制循环无法满足高跟随 ≤10 ms 的要求，因此固定使用低跟随。
        code = arm.rm_movev_canfd(velocity_m_s, False, 0, 0)
        self._check_code("下发 XY 笛卡尔速度", code)
        self._motion_commanded = True

    def stop_xy(self) -> None:
        """用全零笛卡尔速度终止 XY 透传；失败时退回轨迹缓停。"""
        arm = self._connected_or_none()
        if arm is None:
            return
        code = arm.rm_movev_canfd([0.0] * 6, False, 0, 0)
        if code == 0:
            self._motion_commanded = False
            return
        fallback_code = arm.rm_set_arm_slow_stop()
        if fallback_code != 0:
            raise RealManSDKError(
                f"停止 XY 失败：{self._format_error('零速度指令', code)}；"
                f"{self._format_error('轨迹缓停', fallback_code)}"
            )
        self._motion_commanded = False

    def stop_all(self) -> None:
        """终止当前轨迹；不会自动触发或解除控制器的锁存急停。"""
        arm = self._connected_or_none()
        if arm is None:
            return
        if not self._motion_commanded:
            return
        code = arm.rm_set_arm_stop()
        self._check_code("停止机械臂全部运动", code)
        self._last_planned_motion_at = None
        self._last_planned_target_pose_mm = None
        self._motion_commanded = False

    def get_current_pose(self) -> Sequence[float]:
        """返回 ``[x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]``。"""
        arm = self._require_connected()
        code, state = arm.rm_get_current_arm_state()
        self._check_code("读取当前末端位姿", code)
        pose = state.get("pose") if isinstance(state, dict) else None
        if not isinstance(pose, list) or len(pose) != 6:
            raise RealManSDKError(f"机械臂返回无效末端位姿：{pose!r}")
        return [
            float(pose[0]) * 1000.0,
            float(pose[1]) * 1000.0,
            float(pose[2]) * 1000.0,
            float(pose[3]),
            float(pose[4]),
            float(pose[5]),
        ]

    def get_current_work_frame_name(self) -> str:
        """读取当前工作坐标系名称，用于阻止标定与运行坐标系不一致。"""
        name, _ = self._get_current_work_frame()
        return name

    def get_current_work_frame_pose(self) -> Sequence[float]:
        """返回当前工作坐标系相对基坐标系的 SDK 位姿（m/rad）。"""
        _, pose = self._get_current_work_frame()
        return pose

    def _get_current_work_frame(self) -> tuple[str, list[float]]:
        arm = self._require_connected()
        code, frame = arm.rm_get_current_work_frame()
        self._check_code("读取当前工作坐标系", code)
        name = str(frame.get("name", "")).strip("\x00 ") if isinstance(frame, dict) else ""
        pose = frame.get("pose") if isinstance(frame, dict) else None
        if not name or not isinstance(pose, list) or len(pose) != 6:
            raise RealManSDKError(f"机械臂返回无效工作坐标系：{frame!r}")
        values = [float(value) for value in pose]
        if not all(math.isfinite(value) for value in values):
            raise RealManSDKError(f"机械臂返回非有限工作坐标系位姿：{pose!r}")
        return name, values

    def is_moving(self) -> bool:
        """通过当前轨迹类型判断规划运动是否仍在执行。"""
        arm = self._require_connected()
        trajectory = arm.rm_get_arm_current_trajectory()
        code = int(trajectory.get("return_code", -3))
        self._check_code("读取当前运动轨迹", code)
        return int(trajectory.get("trajectory_type", 0)) != 0

    def wait_until_stop(self, timeout_s: float | None = None) -> None:
        """等待规划运动实际到达本次目标位姿，超时则停止并抛出异常。

        不能只依赖 ``trajectory_type``：部分控制器在 ``rm_movel`` 刚受理时会
        暂时返回未运动，旧逻辑因此会过早返回并导致状态机在 Z 尚未下降时吸液。
        """
        timeout = self.config.motion_timeout_s if timeout_s is None else timeout_s
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        target_pose = self._last_planned_target_pose_mm
        while True:
            moving = self.is_moving()
            current_pose = self.get_current_pose()
            reached_target = target_pose is not None and self._is_pose_reached(
                current_pose,
                target_pose,
            )
            if not moving and reached_target:
                self._last_planned_motion_at = None
                self._last_planned_target_pose_mm = None
                self._motion_commanded = False
                return
            if deadline is not None and time.monotonic() >= deadline:
                self.stop_all()
                if target_pose is None:
                    raise TimeoutError(f"等待机械臂停止超时：{timeout} 秒")
                raise TimeoutError(
                    f"等待机械臂到达规划目标超时：{timeout} 秒；"
                    f"target={list(target_pose)!r}，current={list(current_pose)!r}，moving={moving}"
                )
            time.sleep(0.05)

    def _move_joint_to_pose(self, pose: Sequence[float], speed_percent: int, operation: str) -> None:
        arm = self._require_connected()
        target_pose = self._validate_pose(pose)
        sdk_pose = self._to_sdk_pose(target_pose)
        code = arm.rm_movej_p(sdk_pose, self._percent(speed_percent), 0, 0, 0)
        self._check_code(operation, code)
        self._last_planned_motion_at = time.monotonic()
        self._last_planned_target_pose_mm = tuple(target_pose)
        self._motion_commanded = True

    def _move_linear(self, pose: Sequence[float], speed_percent: int, operation: str) -> None:
        arm = self._require_connected()
        target_pose = self._validate_pose(pose)
        sdk_pose = self._to_sdk_pose(target_pose)
        code = arm.rm_movel(sdk_pose, self._percent(speed_percent), 0, 0, 0)
        self._check_code(operation, code)
        self._last_planned_motion_at = time.monotonic()
        self._last_planned_target_pose_mm = tuple(target_pose)
        self._motion_commanded = True

    def _speed_percent(self, speed_mm_s: float) -> int:
        if not math.isfinite(speed_mm_s) or speed_mm_s <= 0:
            raise ValueError("直线运动速度必须是大于 0 的有限 mm/s 数值")
        max_speed = self._max_line_speed_m_s
        if max_speed is None or max_speed <= 0:
            raise RealManSDKError("尚未读取机械臂最大线速度")
        ratio = math.ceil(float(speed_mm_s) / (max_speed * 1000.0) * 100.0)
        return self._percent(ratio)

    @staticmethod
    def _to_sdk_pose(pose: Sequence[float]) -> list[float]:
        values = RealRobot._validate_pose(pose)
        return [values[0] / 1000.0, values[1] / 1000.0, values[2] / 1000.0, *values[3:]]

    @staticmethod
    def _validate_pose(pose: Sequence[float]) -> list[float]:
        """Return one finite engineering-unit pose in mm/rad."""
        values = [float(value) for value in pose]
        if len(values) != 6:
            raise ValueError("机械臂位姿必须包含 6 个元素：[x_mm,y_mm,z_mm,rx,ry,rz]")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("机械臂位姿包含 NaN 或无穷大")
        return values

    @classmethod
    def _is_pose_reached(cls, current: Sequence[float], target: Sequence[float]) -> bool:
        """Check Cartesian feedback against an engineering-unit target pose."""
        current_values = cls._validate_pose(current)
        target_values = cls._validate_pose(target)
        position_error_mm = math.dist(current_values[:3], target_values[:3])
        orientation_errors = [
            abs((actual - expected + math.pi) % (2.0 * math.pi) - math.pi)
            for actual, expected in zip(current_values[3:], target_values[3:])
        ]
        return (
            position_error_mm <= cls._POSITION_REACHED_TOLERANCE_MM
            and max(orientation_errors, default=0.0) <= cls._ORIENTATION_REACHED_TOLERANCE_RAD
        )

    @staticmethod
    def _percent(value: int) -> int:
        return min(100, max(1, int(value)))

    @staticmethod
    def _validate_finite(*values: float) -> None:
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("机械臂运动参数包含 NaN 或无穷大")

    def _connected_or_none(self) -> Any | None:
        with self._connection_lock:
            return self._arm

    def _require_connected(self) -> Any:
        arm = self._connected_or_none()
        if arm is None:
            raise RealManSDKError("睿尔曼机械臂尚未连接")
        return arm

    def _require_z_motion_confirmed(self) -> None:
        if not self.config.z_motion_confirmed:
            raise RealManSDKError(
                "Z 运动参数尚未确认：请实际测量 observe_z_mm、pick_z_mm、safe_z_mm，"
                "核对 z_mode 后把 z_motion_confirmed 改为 true"
            )

    @staticmethod
    def _format_error(operation: str, code: int) -> str:
        description = _SDK_STATUS.get(int(code), "未知 SDK 状态码")
        return f"{operation}失败（code={code}：{description}）"

    @classmethod
    def _check_code(cls, operation: str, code: int) -> None:
        if int(code) != 0:
            raise RealManSDKError(cls._format_error(operation, int(code)))
