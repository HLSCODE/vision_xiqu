"""Background robot-control worker used by the Qt interface."""

from __future__ import annotations

import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from app.calibration_data import (
    CalibrationMetadata,
    file_sha256,
    load_calibration_array,
    require_matching_context,
    validate_metadata_for_config,
)
from app.config import AppConfig
from app.controller import SuctionRobotController
from app.logging_utils import SessionLogger
from app.state import SystemState
from camera.opencv_camera import OpenCVCamera
from control.ibvs import IBVSController
from control.safety import SafetyManager
from robot.realman_robot import RealRobot
from suction.adp_suction import RealSuctionController
from vision.detector import HSVObjectDetector


TERMINAL_STATES = {
    SystemState.FINISHED,
    SystemState.ERROR,
    SystemState.EMERGENCY_STOP,
}


def load_runtime_calibration(config: AppConfig) -> tuple[np.ndarray, np.ndarray, CalibrationMetadata]:
    """加载 GUI 自动吸取必需的 IBVS 矩阵和吸管轴线参考像素。"""
    servo_path = config.path(config.ibvs.servo_A_path)
    reference_path = config.path(config.suction.suction_ref_path)
    matrix, servo_metadata = load_calibration_array(servo_path, "servo_A")
    reference, reference_metadata = load_calibration_array(reference_path, "suction_ref")
    if matrix.shape != (2, 2) or reference.shape != (2,):
        raise ValueError("标定文件尺寸错误：servo_A 必须为 (2,2)，suction_ref 必须为 (2,)")
    validate_metadata_for_config(servo_metadata, config)
    validate_metadata_for_config(reference_metadata, config)
    require_matching_context(servo_metadata, reference_metadata)
    current_servo_hash = file_sha256(servo_path)
    if reference_metadata.servo_A_sha256 != current_servo_hash:
        raise ValueError("suction_ref 对应的 servo_A 已被替换，请重新执行吸管参考点标定")
    return matrix.astype(np.float64), reference.astype(np.float64), servo_metadata


class ControlWorker(QThread):
    """在后台线程运行状态机，避免机械臂动作阻塞 Qt 界面。"""

    state_changed = pyqtSignal(str, str)
    log_message = pyqtSignal(str)
    control_finished = pyqtSignal(str, int)
    control_failed = pyqtSignal(str)

    def __init__(self, config: AppConfig, camera: OpenCVCamera, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.camera = camera
        self._controller: SuctionRobotController | None = None
        self._robot: RealRobot | None = None
        self._suction: RealSuctionController | None = None

    def request_emergency_stop(self) -> None:
        """请求控制循环退出，并尽快向硬件发送停止命令。"""
        self.requestInterruption()
        # 先直接调用硬件停止，处理状态机可能正阻塞在厂商 SDK 等待中的情况。
        if self._robot is not None:
            try:
                self._robot.stop_all()
            except Exception:
                pass
        if self._suction is not None:
            try:
                self._suction.off()
            except Exception:
                pass

    def run(self) -> None:
        session: SessionLogger | None = None
        failure: str | None = None
        final_state = SystemState.ERROR
        pick_count = 0
        try:
            session = SessionLogger(self.config.path("logs"), self.config.system.save_csv)
            self.log_message.emit("正在加载 servo_A 和吸管轴线参考像素标定文件……")
            servo_A, suction_ref, calibration_metadata = load_runtime_calibration(self.config)

            detector = HSVObjectDetector(
                self.config.vision,
                intrinsic_path=self.config.path(self.config.camera.intrinsic_path),
                enable_undistort=self.config.camera.enable_undistort,
                expected_image_size=(self.config.camera.width, self.config.camera.height),
            )
            ibvs = IBVSController(servo_A, self.config.ibvs)
            self._robot = RealRobot(self.config.robot)
            self._suction = RealSuctionController(self.config.suction)
            self.log_message.emit(f"标定矩阵条件数：{ibvs.condition_number:.3f}")
            # 先构造直接 IBVS 控制器；配置错误时不建立机械臂连接、更不会运动。
            self._controller = SuctionRobotController(
                self.config,
                self.camera,
                detector,
                SafetyManager(self._robot, self.config.safety),
                self._suction,
                ibvs,
                suction_ref,
                session,
            )
            self.log_message.emit("直接 IBVS 参数校验通过，正在连接ADP吸液枪……")
            self._suction.connect()
            self.log_message.emit("ADP吸液枪串口连接成功，正在连接机械臂……")
            self._robot.connect()
            work_frame_name = self._robot.get_current_work_frame_name()
            work_frame_pose = np.asarray(self._robot.get_current_work_frame_pose(), dtype=np.float64)
            if work_frame_name != calibration_metadata.work_frame_name:
                raise ValueError(
                    f"机械臂当前工作坐标系 {work_frame_name!r} 与标定时 "
                    f"{calibration_metadata.work_frame_name!r} 不一致"
                )
            calibrated_frame_pose = np.asarray(calibration_metadata.work_frame_pose_m_rad, dtype=np.float64)
            if not np.allclose(work_frame_pose, calibrated_frame_pose, atol=1e-5, rtol=0.0):
                raise ValueError(
                    "机械臂当前工作坐标系定义与标定时同名坐标系的位姿不一致，请恢复坐标系或重新标定"
                )
            self.log_message.emit(f"工作坐标系校验通过：{work_frame_name}")

            previous_state: SystemState | None = None
            period_s = 1.0 / max(self.config.system.loop_hz, 1.0)
            while self._controller.state not in TERMINAL_STATES:
                cycle_started = time.monotonic()
                if self.isInterruptionRequested():
                    self._controller.emergency_stop("operator UI stop")
                    break
                self._controller.step()
                if self._controller.state is not previous_state:
                    previous_state = self._controller.state
                    reason = self._controller.last_transition_reason or f"状态切换：{previous_state.name}"
                    self.state_changed.emit(previous_state.name, reason)
                remaining = period_s - (time.monotonic() - cycle_started)
                if remaining > 0:
                    time.sleep(remaining)

            final_state = self._controller.state
            pick_count = self._controller.pick_count
            if final_state is SystemState.ERROR:
                failure = self._controller.last_transition_reason or "控制器进入 ERROR 状态"
        except Exception as exc:
            failure = str(exc)
            if session is not None:
                session.logger.exception("GUI control worker failed: %s", exc)
        finally:
            # 清理函数分别保护，确保一个 SDK 的异常不会阻止其他设备被关闭。
            if self._robot is not None:
                try:
                    self._robot.stop_all()
                except Exception:
                    pass
            if self._suction is not None:
                try:
                    self._suction.off()
                except Exception:
                    pass
                try:
                    self._suction.close()
                except Exception:
                    pass
            if self._robot is not None:
                try:
                    self._robot.disconnect()
                except Exception:
                    pass
            if session is not None:
                session.close()
            self._controller = None

        if failure is not None:
            self.control_failed.emit(failure)
        else:
            self.control_finished.emit(final_state.name, pick_count)
