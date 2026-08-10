"""Background robot-control worker used by the Qt interface."""

from __future__ import annotations

import time
from dataclasses import replace

from PyQt6.QtCore import QThread, pyqtSignal

from app.config import AppConfig
from app.controller import SuctionRobotController
from app.logging_utils import SessionLogger
from app.state import SystemState
from camera.opencv_camera import OpenCVCamera
from control.ibvs import IBVSController
from control.safety import SafetyManager
from main import load_runtime_calibration
from robot.real_robot_template import RealRobot
from suction.real_suction_template import RealSuctionController
from vision.detector import HSVObjectDetector


TERMINAL_STATES = {
    SystemState.FINISHED,
    SystemState.ERROR,
    SystemState.EMERGENCY_STOP,
}


class ControlWorker(QThread):
    """在后台线程运行状态机，避免机械臂动作阻塞 Qt 界面。"""

    state_changed = pyqtSignal(str, str)
    log_message = pyqtSignal(str)
    control_finished = pyqtSignal(str, int)
    control_failed = pyqtSignal(str)

    def __init__(self, config: AppConfig, camera: OpenCVCamera, parent=None) -> None:
        super().__init__(parent)
        # GUI 使用 Qt QLabel 显示图像，因此关闭控制器内部的 cv2.imshow。
        self.config = replace(
            config,
            system=replace(config.system, show_debug=False, show_mask=False),
        )
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
            self.log_message.emit("正在加载 servo_A 和吸盘参考像素标定文件……")
            servo_A, suction_ref = load_runtime_calibration(self.config)

            detector = HSVObjectDetector(
                self.config.vision,
                intrinsic_path=self.config.path(self.config.camera.intrinsic_path),
                enable_undistort=self.config.camera.enable_undistort,
            )
            ibvs = IBVSController(servo_A, self.config.ibvs)
            self._robot = RealRobot()
            self._suction = RealSuctionController()
            self.log_message.emit(f"标定矩阵条件数：{ibvs.condition_number:.3f}")
            self.log_message.emit("正在连接机械臂……")
            self._robot.connect()

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
                    self.state_changed.emit(previous_state.name, f"状态切换：{previous_state.name}")
                remaining = period_s - (time.monotonic() - cycle_started)
                if remaining > 0:
                    time.sleep(remaining)

            final_state = self._controller.state
            pick_count = self._controller.pick_count
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
