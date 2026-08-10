"""Eye-in-Hand 吸取任务的显式、安全状态机。

本文件只负责任务编排，不直接调用任何厂商 SDK：

1. 在固定观察位完成 HSV 全局检测与像素尺寸筛选；
2. 选定一个目标后，仅用最近邻规则跟踪该目标并进行 2D IBVS；
3. 对准完成后停止 XY 视觉控制，仅执行 Z 下降、吸取、抬升；
4. 目标丢失、超时或安全异常时，停止运动并回到观察位重新检测。

机械臂动作均经 ``SafetyManager`` 下发，吸盘操作均经 ``SuctionController``
下发，因此算法层不依赖具体机械臂、相机或 GPIO/PLC SDK。
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from app.config import AppConfig
from app.logging_utils import SessionLogger
from app.state import SystemState
from camera.opencv_camera import OpenCVCamera
from control.ibvs import IBVSController
from control.safety import SafetyManager, SafetyViolation
from suction.base import SuctionController
from vision.detector import HSVObjectDetector
from vision.models import DetectedObject
from vision.tracker import NearestNeighborTracker
from vision.visualization import draw_debug_overlay


class SuctionRobotController:
    """逐个目标执行“观察→对准→吸取→回观察位”的状态机。

    关键约束：

    - ``GLOBAL_DETECT`` 仅在固定观察位进行尺寸筛选；
    - 一旦进入 ``ALIGN_IBVS``，目标已经锁定，不允许因为其他目标更近而切换；
    - ``DESCEND`` 前必须连续多帧满足像素容差，防止检测噪声触发误吸；
    - 任意异常都必须停止机械臂并关闭吸盘。

    Args:
        config: 所有运行参数，包括视觉、IBVS、Z 轴与安全限制。
        camera: 返回 BGR 图像帧的相机适配器。
        detector: HSV 轮廓检测器。
        safety: 机器人动作的唯一安全入口。
        suction: 吸盘开关适配器。
        ibvs: 将像素误差转换为 XY 速度的控制器。
        suction_ref_px: 吸盘轴线对准目标时的图像像素坐标 ``[u*, v*]``。
        session: 文本日志与可选 CSV 记录器。
    """

    def __init__(
        self,
        config: AppConfig,
        camera: OpenCVCamera,
        detector: HSVObjectDetector,
        safety: SafetyManager,
        suction: SuctionController,
        ibvs: IBVSController,
        suction_ref_px: np.ndarray,
        session: SessionLogger,
    ) -> None:
        self.config = config
        self.camera = camera
        self.detector = detector
        self.safety = safety
        self.suction = suction
        self.ibvs = ibvs
        # 吸盘参考像素来自标定文件，不能用图像中心替代。
        # 该坐标在固定 IBVS 高度下有效，单位为 px，格式为 [u*, v*]。
        self.suction_ref_px = np.asarray(suction_ref_px, dtype=np.float64)
        self.tracker = NearestNeighborTracker(config.tracking.max_distance_px)
        self.session = session
        self.logger: logging.Logger = session.logger
        self.state = SystemState.INIT
        # target/last_center 只保存当前锁定目标，不维护长期的多目标 ID。
        # 本系统的目标静止、IBVS 距离短，因此只需简单最近邻关联即可。
        self.target: DetectedObject | None = None
        # last_center_px 是下一帧最近邻匹配的参考点；一旦对象丢失，不从其他候选中强行切换。
        self.last_center_px: np.ndarray | None = None
        # candidate 只在“全局检测→锁定目标”之间短暂保存，尚未进入 IBVS。
        self._candidate: DetectedObject | None = None
        # stable_frames 防止单帧偶然落入容差即开始 Z 下降。
        self._stable_frames = 0
        # lost_frames 允许短时图像漏检，超过上限后才进入 RECOVER。
        self._lost_frames = 0
        # growth_frames 用于发现误差持续扩大（例如 A 矩阵方向错误或目标关联错误）。
        self._growth_frames = 0
        self._previous_error_norm: float | None = None
        self._last_control_time = time.monotonic()
        self._suction_started_at: float | None = None
        self.pick_count = 0

    def transition(self, new_state: SystemState, reason: str = "") -> None:
        """切换状态并记录原因，便于复盘机械臂任务过程。

        状态切换集中在此处记录，日志可直接还原一次吸取失败是由于目标丢失、
        对准超时，还是硬件/SDK 异常。
        """
        self.logger.info("STATE %s -> %s %s", self.state.name, new_state.name, reason)
        self.state = new_state

    def run(self, max_steps: int | None = None) -> SystemState:
        """以固定频率推进状态机，直到结束、错误或急停。

        ``max_steps`` 主要用于现场首轮验证的硬上限：超过该次数立即急停，
        避免配置错误时控制循环无限持续。正常生产运行可传入 ``None``。
        """
        steps = 0
        period_s = 1.0 / max(self.config.system.loop_hz, 1.0)
        while self.state not in {SystemState.FINISHED, SystemState.ERROR, SystemState.EMERGENCY_STOP}:
            started = time.monotonic()
            self.step()
            steps += 1
            if max_steps is not None and steps >= max_steps:
                self.logger.warning("Reached max_steps=%d; issuing emergency stop", max_steps)
                self.emergency_stop("maximum steps reached")
                break
            remaining = period_s - (time.monotonic() - started)
            if remaining > 0:
                # 保持近似固定控制周期，IBVS 的速度和加速度限制依赖这个时间间隔。
                time.sleep(remaining)
        return self.state

    def step(self) -> None:
        """执行当前状态的一次处理；本函数不负责无限循环。

        每次调用最多完成一个离散状态动作或一次图像闭环控制。将状态函数拆开，
        可保证下降、吸取与恢复动作不会和持续 IBVS XY 速度混在同一个代码分支中。
        """
        try:
            # 每个状态有独立处理函数，避免把检测、对准和吸取堆在一个 while 循环中。
            handlers = {
                SystemState.INIT: self._handle_init,
                SystemState.MOVE_TO_OBSERVE: self._handle_move_to_observe,
                SystemState.GLOBAL_DETECT: self._handle_global_detect,
                SystemState.SELECT_TARGET: self._handle_select_target,
                SystemState.ALIGN_IBVS: self._handle_align_ibvs,
                SystemState.DESCEND: self._handle_descend,
                SystemState.SUCTION: self._handle_suction,
                SystemState.LIFT: self._handle_lift,
                SystemState.RETURN_OBSERVE: self._handle_return_observe,
                SystemState.RECOVER: self._handle_recover,
            }
            handler = handlers.get(self.state)
            if handler is not None:
                handler()
        except SafetyViolation as exc:
            # 安全限制触发时可尝试回观察位重新开始，不直接继续当前对准动作。
            self.logger.error("Safety violation: %s", exc)
            self.suction.off()
            self.transition(SystemState.RECOVER, str(exc))
        except Exception as exc:
            # 未预期的相机/机械臂/吸盘 SDK 异常不能恢复性地继续运动，必须进入 ERROR。
            self.logger.exception("Unhandled controller error: %s", exc)
            self.safety.stop_all()
            self.suction.off()
            self.transition(SystemState.ERROR, str(exc))

    def emergency_stop(self, reason: str) -> None:
        """由操作员、Ctrl+C 或步数限制触发的急停。

        ``stop_all`` 必须先执行，确保已下发的 XY 速度不再继续；随后关闭吸盘。
        急停为终止状态，不会像 ``RECOVER`` 一样自动再次执行全局检测。
        """
        self.logger.warning("EMERGENCY STOP: %s", reason)
        self.safety.stop_all()
        self.suction.off()
        self.transition(SystemState.EMERGENCY_STOP, reason)

    def _get_detection(self) -> tuple[np.ndarray, list[DetectedObject], list[DetectedObject], np.ndarray] | None:
        """读取一帧并完成 HSV 检测。

        Returns:
            ``(frame, objects, valid, mask)``。``objects`` 包含全部通过最小面积
            的橙黄色轮廓；``valid`` 额外满足像素尺寸阈值，仅供全局选目标使用。
            相机单帧失败时返回 ``None``，连续失败次数由安全层负责判断。
        """
        frame = self.camera.get_frame()
        # 连续断流由 SafetyManager 计数并触发停止，单帧失败不会继续使用旧图像运动。
        self.safety.report_camera_frame(frame is not None)
        if frame is None:
            return None
        result = self.detector.detect(frame)
        valid = self.detector.valid_objects(result.objects)
        return frame, result.objects, valid, result.mask

    def _show(self, frame: np.ndarray, objects: list[DetectedObject], valid: list[DetectedObject], mask: np.ndarray,
              error: np.ndarray | None = None, velocity: np.ndarray | None = None) -> None:
        """按配置显示实时调试叠加层，并监听操作员键盘急停。"""
        if not self.config.system.show_debug:
            return
        dt = max(1e-6, time.monotonic() - self._last_control_time)
        overlay = draw_debug_overlay(frame, objects, valid, self.suction_ref_px, self.state, self.target, error, velocity, 1.0 / dt)
        cv2.imshow("Robot Suction IBVS", overlay)
        if self.config.system.show_mask:
            cv2.imshow("HSV mask", mask)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self.emergency_stop("operator keyboard stop")

    def _handle_init(self) -> None:
        """初始状态：不运动，只进入观察位流程。"""
        self.transition(SystemState.MOVE_TO_OBSERVE, "initialization complete")

    def _handle_move_to_observe(self) -> None:
        """移动至已教示的固定观察位，再开始尺寸相关的全局检测。"""
        # 观察位/高度固定，是用像素而非毫米进行尺寸筛选的前提。
        self.safety.move_to_observe_pose()
        self.transition(SystemState.GLOBAL_DETECT, "at fixed observation pose")

    def _handle_global_detect(self) -> None:
        """在观察位检测全部目标，并选择一个尺寸合格的候选目标。"""
        data = self._get_detection()
        if data is None:
            return
        frame, objects, valid, mask = data
        self._show(frame, objects, valid, mask)
        if not valid:
            # 全局观察位没有合格尺寸目标，当前批次任务完成。
            self.transition(SystemState.FINISHED, "no size-eligible targets at observation pose")
            return
        # 尺寸合格的候选中，优先选择距吸盘参考像素最近的一个。
        # 该规则只决定“先吸哪个”，不会改变尺寸是否允许吸取的判断。
        self._candidate = self.tracker.select_nearest_reference(valid, self.suction_ref_px)
        self.transition(SystemState.SELECT_TARGET, f"selected candidate #{self._candidate.index}")

    def _handle_select_target(self) -> None:
        """锁定全局检测选出的候选目标，并初始化一次新的 IBVS 尝试。"""
        if self._candidate is None:
            self.transition(SystemState.RECOVER, "candidate disappeared before locking")
            return
        self.target = self._candidate
        self.last_center_px = self.target.center.copy()
        # 进入 IBVS 后锁定该对象；后续不能改选其他候选目标。
        self._candidate = None
        self._stable_frames = 0
        self._lost_frames = 0
        self._growth_frames = 0
        self._previous_error_norm = None
        # 记录控制周期起点，使 IBVS 的速度变化率限制使用真实 dt。
        self._last_control_time = time.monotonic()
        # 上一个目标的速度不能带入下一个目标，否则切换瞬间可能产生不必要的惯性命令。
        self.ibvs.reset()
        # 重置安全层的累计 XY 行程、相机失败计数与对准开始时间。
        self.safety.begin_alignment()
        self.transition(SystemState.ALIGN_IBVS, f"locked target #{self.target.index}")

    def _handle_align_ibvs(self) -> None:
        """针对当前锁定目标执行一帧 IBVS 闭环控制。

        本状态只处理 XY：相机检测目标中心，IBVS 计算 ``vx, vy``，安全层下发
        速度。Z 轴在此阶段保持不动，直到连续稳定帧确认对准成功。
        """
        if self.last_center_px is None:
            self.transition(SystemState.RECOVER, "missing locked target centre")
            return
        if self.safety.alignment_elapsed_s() > self.config.ibvs.max_align_time_s:
            # 对准超时后必须先停止 XY，避免机械臂继续沿最后速度运动。
            self.safety.stop_xy()
            self.transition(SystemState.RECOVER, "IBVS alignment timeout")
            return
        data = self._get_detection()
        if data is None:
            return
        frame, objects, valid, mask = data
        # IBVS 期间不做尺寸筛选或重新选目标，只对上一帧中心做最近邻关联。
        association = self.tracker.associate(self.last_center_px, objects)
        if association.target is None:
            # 当前帧找不到与上帧足够接近的轮廓：不能改吸另一个可见对象。
            self._lost_frames += 1
            self.safety.stop_xy()
            self._show(frame, objects, valid, mask)
            if self._lost_frames >= self.config.tracking.max_lost_frames:
                self.transition(SystemState.RECOVER, "locked target lost")
            return
        self._lost_frames = 0
        self.target = association.target
        self.last_center_px = self.target.center.copy()
        now = time.monotonic()
        # compute() 输出单位为 mm/s；dt 用于其中的加速度/速度变化率限制。
        command = self.ibvs.compute(self.target.center, self.suction_ref_px, now - self._last_control_time)
        self._last_control_time = now
        if self._previous_error_norm is not None and command.error_norm_px > self._previous_error_norm + 0.5:
            # 误差连续增长通常意味着映射方向、目标关联或机械臂执行存在异常。
            self._growth_frames += 1
        else:
            self._growth_frames = 0
        self._previous_error_norm = command.error_norm_px
        if self._growth_frames >= self.config.ibvs.error_growth_frames:
            self.safety.stop_xy()
            self.transition(SystemState.RECOVER, "IBVS error increased repeatedly")
            return
        if np.all(np.abs(command.error_px) < self.config.ibvs.pixel_tolerance):
            # 必须连续多帧满足容差，避免由单帧噪声触发下降。
            self._stable_frames += 1
        else:
            self._stable_frames = 0
        self.safety.set_xy_velocity(command.velocity_mm_s)
        # CSV 用于事后分析误差收敛、速度饱和与目标关联距离。
        self.session.record(
            state=self.state.name, u=self.target.center[0], v=self.target.center[1],
            u_ref=self.suction_ref_px[0], v_ref=self.suction_ref_px[1],
            error_u=command.error_px[0], error_v=command.error_px[1],
            vx=command.velocity_mm_s[0], vy=command.velocity_mm_s[1],
            size_px=self.target.size_px, tracking_distance=association.distance_px,
        )
        self._show(frame, objects, valid, mask, command.error_px, command.velocity_mm_s)
        if self._stable_frames >= self.config.ibvs.stable_frames:
            self.safety.stop_xy()
            self.transition(SystemState.DESCEND, "pixel tolerance stable across frames")

    def _handle_descend(self) -> None:
        """XY 已对准后，按配置执行绝对或相对 Z 轴下降。"""
        # 对准后停止全部 XY 视觉控制；Z 轴不参与 IBVS，而是执行教示高度。
        self.safety.stop_xy()
        if self.config.robot.z_mode == "absolute":
            # 绝对模式直接到达教示的 pick_z_mm。
            self.safety.move_z_absolute(self.config.robot.pick_z_mm, self.config.robot.z_down_speed_mm_s)
        elif self.config.robot.z_mode == "relative":
            # 相对模式以观察高度为基准计算下降距离，避免依赖机器人绝对 Z 原点。
            self.safety.move_z_relative(self.config.robot.pick_z_mm - self.config.robot.observe_z_mm, self.config.robot.z_down_speed_mm_s)
        else:
            raise ValueError("robot.z_mode must be 'absolute' or 'relative'")
        self.transition(SystemState.SUCTION, "reached taught pickup height")

    def _handle_suction(self) -> None:
        """开启吸盘并等待设定保持时间，确保负压建立后再抬升。"""
        if self._suction_started_at is None:
            self.suction.on()
            # 不阻塞主循环，下一周期按保持时间判断是否可以抬升。
            self._suction_started_at = time.monotonic()
            self.logger.info("Suction hold started")
            return
        if time.monotonic() - self._suction_started_at >= self.config.suction.hold_time_s:
            self.pick_count += 1
            self._suction_started_at = None
            self.transition(SystemState.LIFT, f"suction hold complete; picks={self.pick_count}")

    def _handle_lift(self) -> None:
        """将吸取后的目标抬至安全高度，并关闭吸盘。"""
        if self.config.robot.z_mode == "absolute":
            self.safety.move_z_absolute(self.config.robot.safe_z_mm, self.config.robot.z_up_speed_mm_s)
        else:
            self.safety.move_z_relative(self.config.robot.safe_z_mm - self.config.robot.pick_z_mm, self.config.robot.z_up_speed_mm_s)
        self.suction.off()
        self.transition(SystemState.RETURN_OBSERVE, "object lifted")

    def _handle_return_observe(self) -> None:
        """回固定观察位，清除锁定目标，再处理下一批静止目标。"""
        self.safety.move_to_observe_pose()
        self.target = None
        self.last_center_px = None
        self.transition(SystemState.GLOBAL_DETECT, "returned to observe pose")

    def _handle_recover(self) -> None:
        """对准阶段失败后的可恢复处理。

        该状态与 ``ERROR`` 的差异是：它会在完成停止与回观察位后重新检测；
        ``ERROR`` 则终止本次任务并等待人工处理。
        """
        # 目标丢失、超时或安全异常：停止、关吸盘、回观察位，再重新做全局检测。
        self.safety.stop_all()
        self.suction.off()
        self.target = None
        self.last_center_px = None
        self.safety.move_to_observe_pose()
        self.transition(SystemState.GLOBAL_DETECT, "recovery at observe pose")
