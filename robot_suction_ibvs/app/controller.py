"""Eye-in-Hand 吸取任务的显式、安全状态机。

本文件只负责任务编排，不直接调用任何厂商 SDK：

1. 在固定观察位完成 HSV 全局检测与像素尺寸筛选；
2. 选定一个目标后，仅用最近邻规则跟踪该目标，并直接对准到吸管轴线参考点；
3. 目标连续稳定对准后，停止 XY 并执行 Z 下降、吸取、抬升；
4. 目标丢失、超时或安全异常时，停止运动并回到观察位重新检测。

机械臂动作均经 ``SafetyManager`` 下发，吸取操作均经 ``SuctionController``
下发，因此算法层不依赖具体机械臂、相机或 ADP 串口实现。
"""

from __future__ import annotations

import logging
import time

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


class SuctionRobotController:
    """逐个目标执行“观察→对准→吸取→回观察位”的状态机。

    关键约束：

    - ``GLOBAL_DETECT`` 仅在固定观察位进行尺寸筛选；
    - 一旦进入 ``ALIGN_IBVS``，目标已经锁定，不允许因为其他目标更近而切换；
    - 目标仍可见时必须连续多帧到达吸管轴线参考点，之后才允许下降；
    - ``DESCEND`` 前必须停止 XY 视觉控制；
    - 任意异常都必须停止机械臂并结束当前吸取软件状态。

    Args:
        config: 所有运行参数，包括视觉、IBVS、Z 轴与安全限制。
        camera: 返回 BGR 图像帧的相机适配器。
        detector: HSV 轮廓检测器。
        safety: 机器人动作的唯一安全入口。
        suction: 吸取执行器适配器；当前真实实现为 ADP 定量吸液枪。
        ibvs: 将像素误差转换为 XY 速度的控制器。
        suction_ref_px: 吸管轴线正对目标时，目标中心在图像中的参考像素 ``[u*, v*]``。
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
        # 吸管参考像素来自单点标定文件，不能用图像中心替代。
        # 当前硬件构型下，该点仍能看到目标中心，因此可直接闭环到该位置。
        self.suction_ref_px = np.asarray(suction_ref_px, dtype=np.float64)
        if self.suction_ref_px.shape != (2,) or not np.all(np.isfinite(self.suction_ref_px)):
            raise ValueError("suction_ref_px must be a finite [u, v] vector")

        self.tracker = NearestNeighborTracker(config.tracking.max_distance_px)
        self.session = session
        self.logger: logging.Logger = session.logger
        self.state = SystemState.INIT
        self.last_transition_reason = ""
        # target/last_center 只保存当前锁定目标，不维护长期的多目标 ID。
        # 本系统的目标静止、IBVS 距离短，因此只需简单最近邻关联即可。
        self.target: DetectedObject | None = None
        # last_center_px 是下一帧最近邻匹配的参考点；一旦对象丢失，不从其他候选中强行切换。
        self.last_center_px: np.ndarray | None = None
        # candidate 只在“全局检测→锁定目标”之间短暂保存，尚未进入 IBVS。
        self._candidate: DetectedObject | None = None
        # stable_frames 防止单帧偶然落入吸管轴线容差就触发下降。
        self._stable_frames = 0
        # lost_frames 允许短时图像漏检，超过上限后才进入 RECOVER。
        self._lost_frames = 0
        # growth_frames 用于发现误差持续扩大（例如 A 矩阵方向错误或目标关联错误）。
        self._growth_frames = 0
        # 全局观察必须连续多帧为空才结束任务，避免一次曝光波动或轮廓漏检提前完成。
        self._empty_scene_frames = 0
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
        self.last_transition_reason = reason

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
            cleanup_failures = self._best_effort_stop_and_suction_off()
            if cleanup_failures:
                self.transition(SystemState.ERROR, f"{exc}; safety cleanup failed: {'; '.join(cleanup_failures)}")
            else:
                self.transition(SystemState.RECOVER, str(exc))
        except Exception as exc:
            # 未预期的相机/机械臂/ADP 串口异常不能恢复性地继续运动，必须进入 ERROR。
            self.logger.exception("Unhandled controller error: %s", exc)
            cleanup_failures = self._best_effort_stop_and_suction_off()
            reason = str(exc)
            if cleanup_failures:
                reason += "; cleanup failed: " + "; ".join(cleanup_failures)
            self.transition(SystemState.ERROR, reason)

    def _best_effort_stop_and_suction_off(self) -> list[str]:
        """Run independent safety cleanup actions without masking the original failure."""
        failures: list[str] = []
        try:
            self.safety.stop_all()
        except Exception as cleanup_exc:
            failures.append(f"stop_all: {cleanup_exc}")
            self.logger.exception("Failed to stop robot during cleanup")
        try:
            self.suction.off()
        except Exception as cleanup_exc:
            failures.append(f"suction.off: {cleanup_exc}")
            self.logger.exception("Failed to switch suction off during cleanup")
        self._suction_started_at = None
        return failures

    def emergency_stop(self, reason: str) -> None:
        """由操作员、Ctrl+C 或步数限制触发的急停。

        ``stop_all`` 必须先执行，确保已下发的 XY 速度不再继续；随后结束吸取软件状态。
        急停为终止状态，不会像 ``RECOVER`` 一样自动再次执行全局检测。
        """
        self.logger.warning("EMERGENCY STOP: %s", reason)
        cleanup_failures = self._best_effort_stop_and_suction_off()
        if cleanup_failures:
            reason += "; cleanup failed: " + "; ".join(cleanup_failures)
        self.transition(SystemState.EMERGENCY_STOP, reason)

    def _get_detection(self) -> list[DetectedObject] | None:
        """读取一帧并完成 HSV 检测。

        Returns:
            通过最小面积筛选的目标列表。相机单帧失败时返回 ``None``，
            连续失败次数由安全层负责判断。尺寸资格只在全局选目标阶段计算。
        """
        frame = self.camera.get_frame()
        # 连续断流由 SafetyManager 计数并触发停止，单帧失败不会继续使用旧图像运动。
        self.safety.report_camera_frame(frame is not None)
        if frame is None:
            return None
        result = self.detector.detect(frame)
        return result.objects

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
        objects = data
        valid = self.detector.valid_objects(objects)
        if not valid:
            self._empty_scene_frames += 1
            if self._empty_scene_frames >= self.config.tracking.empty_scene_confirm_frames:
                self.transition(
                    SystemState.FINISHED,
                    f"no size-eligible targets for {self._empty_scene_frames} consecutive frames",
                )
            return
        self._empty_scene_frames = 0
        # 尺寸合格的候选中，优先选择距吸管轴线参考点最近的一个。
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

        本状态只处理 XY：相机检测目标中心，IBVS 直接将目标移动到
        ``suction_ref_px``。Z 轴保持不动；目标在此阶段消失仍按真实丢失处理，
        不允许把任意漏检误判为已经到达吸取位置。
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
        objects = data
        # IBVS 期间不做尺寸筛选或重新选目标，只对上一帧中心做最近邻关联。
        association = self.tracker.associate(self.last_center_px, objects)
        if association.target is None:
            # 当前帧找不到与上帧足够接近的轮廓：不能改吸另一个可见对象。
            self._lost_frames += 1
            self.safety.stop_xy()
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
        if command.error_norm_px <= self.config.ibvs.pixel_tolerance:
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
        if self._stable_frames >= self.config.ibvs.stable_frames:
            self.safety.stop_xy()
            self.transition(SystemState.DESCEND, "direct IBVS alignment stable")

    def _handle_descend(self) -> None:
        """XY 已对准后，按配置执行绝对或相对 Z 轴下降。"""
        # 对准后停止全部 XY 视觉控制；Z 轴不参与 IBVS，而是执行教示高度。
        self.safety.stop_xy()
        if self.config.robot.z_mode == "absolute":
            # 绝对模式直接到达教示的 pick_z_mm。
            self.safety.move_z_absolute(self.config.robot.pick_z_mm, self.config.robot.z_down_speed_mm_s)
        elif self.config.robot.z_mode == "relative":
            # 相对模式以观察高度为基准计算下降距离，避免依赖机器人绝对 Z 原点。
            self.safety.move_z_relative(
                self.config.robot.pick_z_mm - self.config.robot.observe_z_mm,
                self.config.robot.z_down_speed_mm_s,
            )
        else:
            raise ValueError("robot.z_mode must be 'absolute' or 'relative'")
        self.transition(SystemState.SUCTION, "reached taught pickup height")

    def _handle_suction(self) -> None:
        """发送一次 ADP 定量吸液命令并等待设定保持时间后再抬升。"""
        if self._suction_started_at is None:
            self.suction.on()
            if not self.suction.is_on():
                raise RuntimeError("ADP吸液命令已发送，但软件命令状态未确认")
            # 不阻塞主循环，下一周期按保持时间判断是否可以抬升。
            self._suction_started_at = time.monotonic()
            self.logger.info("Suction hold started")
            return
        if not self.suction.is_on():
            raise RuntimeError("吸取保持阶段检测到ADP软件命令状态已被停止")
        if time.monotonic() - self._suction_started_at >= self.config.suction.hold_time_s:
            self._suction_started_at = None
            self.transition(SystemState.LIFT, "suction hold complete")

    def _handle_lift(self) -> None:
        """将吸取后的目标抬至安全高度，并结束本轮 ADP 软件保持状态。"""
        if self.config.robot.z_mode == "absolute":
            self.safety.move_z_absolute(self.config.robot.safe_z_mm, self.config.robot.z_up_speed_mm_s)
        else:
            self.safety.move_z_relative(
                self.config.robot.safe_z_mm - self.config.robot.pick_z_mm,
                self.config.robot.z_up_speed_mm_s,
            )
        self.suction.off()
        # 只有抬升和结束本轮吸取状态都完成后才计为一次完整吸取，失败的中途动作不计数。
        self.pick_count += 1
        self.transition(SystemState.RETURN_OBSERVE, f"object lifted; picks={self.pick_count}")

    def _handle_return_observe(self) -> None:
        """回固定观察位，清除锁定目标，再处理下一批静止目标。"""
        self.safety.move_to_observe_pose()
        self.target = None
        self.last_center_px = None
        self._empty_scene_frames = 0
        self.transition(SystemState.GLOBAL_DETECT, "returned to observe pose")

    def _handle_recover(self) -> None:
        """对准阶段失败后的可恢复处理。

        该状态与 ``ERROR`` 的差异是：它会在完成停止与回观察位后重新检测；
        ``ERROR`` 则终止本次任务并等待人工处理。
        """
        # 目标丢失、超时或安全异常：停止、结束吸取状态、回观察位，再重新做全局检测。
        self.safety.stop_all()
        self.suction.off()
        self.target = None
        self.last_center_px = None
        self._empty_scene_frames = 0
        self._suction_started_at = None
        self.safety.move_to_observe_pose()
        self.transition(SystemState.GLOBAL_DETECT, "recovery at observe pose")
