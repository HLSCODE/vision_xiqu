"""用于相机预览和启动吸取任务的 PyQt6 主窗口。"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TypeVar

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from camera.opencv_camera import OpenCVCamera
from ui.control_worker import ControlWorker
from ui.detection_worker import PreviewDetection, PreviewDetectionWorker
from ui.styles import APP_STYLE
from vision.visualization import draw_detection_overlay


WidgetType = TypeVar("WidgetType", bound=QWidget)


def _named(widget: WidgetType, object_name: str) -> WidgetType:
    """显式设置 objectName，兼容 PyQt6 并集中支持 QSS 选择器。"""
    widget.setObjectName(object_name)
    return widget


STATE_LABELS = {
    "INIT": "初始化",
    "MOVE_TO_OBSERVE": "移动到观察位",
    "GLOBAL_DETECT": "全局识别",
    "SELECT_TARGET": "选择目标",
    "ALIGN_IBVS": "预对准视觉伺服",
    "FINAL_XY_APPROACH": "终末 XY 移动",
    "DESCEND": "垂直下降",
    "SUCTION": "吸取保持",
    "LIFT": "垂直抬升",
    "RETURN_OBSERVE": "返回观察位",
    "RECOVER": "安全恢复",
    "FINISHED": "任务完成",
    "ERROR": "运行错误",
    "EMERGENCY_STOP": "已急停",
}


class MainWindow(QMainWindow):
    """显示实时相机画面，并控制一次完整自动识别吸取任务。"""

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.camera = OpenCVCamera(config.camera)
        self.control_worker: ControlWorker | None = None
        self.camera_open = False
        self._last_qimage: QImage | None = None
        self._latest_detection: PreviewDetection | None = None
        self._detection_updated_at = 0.0
        self._frame_count = 0
        self._fps_started = time.monotonic()

        self.setWindowTitle("Eye-in-Hand 小目标自动吸取系统")
        self.setMinimumSize(1080, 700)
        self.resize(1380, 860)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()

        self.detection_worker = PreviewDetectionWorker(config.vision, self)
        self.detection_worker.detections_ready.connect(self._on_detections_ready)
        self.detection_worker.detection_failed.connect(self._on_detection_failed)
        self.detection_worker.start()

        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(50)  # 20 FPS 对操作预览足够，并降低 12MP 图像显示负担。
        self.preview_timer.timeout.connect(self._update_preview)
        QTimer.singleShot(0, self._open_camera)

    def _build_ui(self) -> None:
        root = _named(QWidget(), "root")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        header = _named(QFrame(), "header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 13, 18, 13)
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = _named(QLabel("Eye-in-Hand 小目标自动吸取"), "title")
        subtitle = _named(QLabel("高位视觉对准 · 垂直下降吸取"), "subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box)
        header_layout.addStretch()
        self.status_label = _named(QLabel("正在连接相机"), "statusPill")
        self.status_label.setProperty("level", "idle")
        header_layout.addWidget(self.status_label)
        root_layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        video_panel = _named(QFrame(), "videoPanel")
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(12, 12, 12, 12)
        video_layout.setSpacing(8)
        video_header = QHBoxLayout()
        video_header.addWidget(_named(QLabel("实时相机"), "sectionTitle"))
        video_header.addStretch()
        self.target_info = _named(QLabel("检测 0 · 可吸 0"), "detectionCount")
        self.target_info.setToolTip("绿色为尺寸合格目标，橙色为超过 size_threshold_px 的目标")
        video_header.addWidget(self.target_info)
        self.video_info = _named(QLabel("-- × --  |  -- FPS"), "muted")
        video_header.addWidget(self.video_info)
        video_layout.addLayout(video_header)
        self.video_label = _named(QLabel("正在打开相机……"), "video")
        self.video_label.setAccessibleName("实时相机画面")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_layout.addWidget(self.video_label, 1)
        splitter.addWidget(video_panel)

        panel = _named(QFrame(), "panel")
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(410)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 16, 16, 16)
        panel_layout.setSpacing(12)
        panel_layout.addWidget(_named(QLabel("任务控制"), "sectionTitle"))

        self.start_button = _named(QPushButton("开始识别吸取"), "startButton")
        self.start_button.setAccessibleName("开始识别吸取")
        self.start_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._start_control)
        panel_layout.addWidget(self.start_button)

        self.stop_button = _named(QPushButton("停止 / 急停"), "stopButton")
        self.stop_button.setAccessibleName("停止和急停")
        self.stop_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserStop))
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_control)
        panel_layout.addWidget(self.stop_button)

        self.reconnect_button = _named(QPushButton("重新连接相机"), "secondaryButton")
        self.reconnect_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.reconnect_button.clicked.connect(self._reconnect_camera)
        panel_layout.addWidget(self.reconnect_button)

        panel_layout.addWidget(_named(QLabel("运行状态"), "sectionTitle"))
        self.task_state = _named(QLabel("等待相机连接"), "muted")
        self.task_state.setWordWrap(True)
        panel_layout.addWidget(self.task_state)

        panel_layout.addWidget(_named(QLabel("运行日志"), "sectionTitle"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(300)
        panel_layout.addWidget(self.log_view, 1)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([980, 360])
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def _set_status(self, text: str, level: str) -> None:
        self.status_label.setText(text)
        self.status_label.setProperty("level", level)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _append_log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{stamp}] {message}")

    def _open_camera(self) -> None:
        self._set_status("正在连接相机", "idle")
        self.task_state.setText("正在打开 OpenCV 相机……")
        try:
            self.camera.open()
        except Exception as exc:
            self.camera_open = False
            self.video_label.setText("相机连接失败\n请检查设备编号、权限和占用情况")
            self.task_state.setText(str(exc))
            self._set_status("相机异常", "error")
            self._append_log(f"相机连接失败：{exc}")
            return
        self.camera_open = True
        self.preview_timer.start()
        width, height = self.camera.get_resolution()
        self._set_status("相机已连接", "ready")
        self.task_state.setText("系统待机，可开始识别吸取")
        self.start_button.setEnabled(True)
        self._append_log(f"相机已连接，采集分辨率 {width} × {height}")

    def _reconnect_camera(self) -> None:
        if self.control_worker is not None and self.control_worker.isRunning():
            QMessageBox.warning(self, "任务运行中", "请先停止吸取任务，再重新连接相机。")
            return
        self.preview_timer.stop()
        self.camera.close()
        self.camera_open = False
        self._clear_detections()
        self.start_button.setEnabled(False)
        self.video_label.setText("正在重新连接相机……")
        QTimer.singleShot(100, self._open_camera)

    def _update_preview(self) -> None:
        if not self.camera_open:
            return
        try:
            frame = self.camera.get_preview_frame()
        except Exception as exc:
            self.preview_timer.stop()
            self.camera_open = False
            self.start_button.setEnabled(False)
            self._set_status("相机断开", "error")
            self._append_log(f"预览失败：{exc}")
            return
        if frame is None:
            return
        self.detection_worker.submit_frame(frame, self.camera.get_resolution())

        detection = self._latest_detection
        if (
            detection is not None
            and detection.frame_width == frame.shape[1]
            and detection.frame_height == frame.shape[0]
            and time.monotonic() - self._detection_updated_at <= 1.0
        ):
            display_frame = draw_detection_overlay(
                frame,
                detection.objects,
                detection.valid_indices,
            )
        else:
            display_frame = frame

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        self._last_qimage = image
        self._display_qimage(image)

        self._frame_count += 1
        elapsed = time.monotonic() - self._fps_started
        if elapsed >= 1.0:
            fps = self._frame_count / elapsed
            capture_width, capture_height = self.camera.get_resolution()
            self.video_info.setText(f"{capture_width} × {capture_height}  |  {fps:.1f} FPS")
            self._frame_count = 0
            self._fps_started = time.monotonic()

    def _on_detections_ready(self, detection: PreviewDetection) -> None:
        """接收后台检测结果；实际叠加绘制在下一次预览刷新中完成。"""
        self._latest_detection = detection
        self._detection_updated_at = time.monotonic()
        total = len(detection.objects)
        self.target_info.setText(f"检测 {total} · 可吸 {detection.valid_count}")

    def _on_detection_failed(self, error: str) -> None:
        self._clear_detections()
        self._append_log(f"预览目标检测失败：{error}")

    def _clear_detections(self) -> None:
        self._latest_detection = None
        self._detection_updated_at = 0.0
        self.target_info.setText("检测 0 · 可吸 0")

    def _display_qimage(self, image: QImage) -> None:
        target = self.video_label.size()
        pixmap = QPixmap.fromImage(image).scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt method name
        super().resizeEvent(event)
        if self._last_qimage is not None:
            self._display_qimage(self._last_qimage)

    def _start_control(self) -> None:
        if not self.camera_open:
            QMessageBox.warning(self, "相机未连接", "请先连接相机。")
            return
        if self.control_worker is not None and self.control_worker.isRunning():
            return
        self.control_worker = ControlWorker(self.config, self.camera, self)
        self.control_worker.state_changed.connect(self._on_state_changed)
        self.control_worker.log_message.connect(self._append_log)
        self.control_worker.control_finished.connect(self._on_control_finished)
        self.control_worker.control_failed.connect(self._on_control_failed)
        self.control_worker.finished.connect(self._on_worker_stopped)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.reconnect_button.setEnabled(False)
        self._set_status("任务运行中", "running")
        self.task_state.setText("正在初始化识别吸取任务……")
        self._append_log("操作员启动自动识别吸取")
        self.control_worker.start()

    def _stop_control(self) -> None:
        worker = self.control_worker
        if worker is None or not worker.isRunning():
            return
        self.stop_button.setEnabled(False)
        self.task_state.setText("正在执行安全停止……")
        self._append_log("操作员请求停止 / 急停")
        worker.request_emergency_stop()

    def _on_state_changed(self, state: str, message: str) -> None:
        label = STATE_LABELS.get(state, state)
        self.task_state.setText(label)
        self._append_log(f"{message}（{label}）")

    def _on_control_finished(self, state: str, pick_count: int) -> None:
        label = STATE_LABELS.get(state, state)
        level = "ready" if state == "FINISHED" else "error"
        self._set_status(label, level)
        self.task_state.setText(f"{label}，本次吸取数量：{pick_count}")
        self._append_log(f"任务结束：{label}，吸取 {pick_count} 个目标")

    def _on_control_failed(self, error: str) -> None:
        self._set_status("任务错误", "error")
        self.task_state.setText(error)
        self._append_log(f"任务启动或运行失败：{error}")
        QMessageBox.critical(self, "识别吸取失败", error)

    def _on_worker_stopped(self) -> None:
        worker = self.control_worker
        if worker is not None:
            worker.deleteLater()
        self.control_worker = None
        self.start_button.setEnabled(self.camera_open)
        self.stop_button.setEnabled(False)
        self.reconnect_button.setEnabled(True)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt method name
        worker = self.control_worker
        if worker is not None and worker.isRunning():
            worker.request_emergency_stop()
            if not worker.wait(5000):
                # 不销毁仍在运行的 QThread；等待厂商 SDK 从阻塞调用中退出后再关闭。
                self._append_log("机械臂控制线程尚未退出，窗口暂不能关闭")
                QMessageBox.warning(self, "正在安全停止", "机械臂控制仍在停止中，请稍后再次关闭窗口。")
                event.ignore()
                return
        self.preview_timer.stop()
        self.detection_worker.stop()
        if not self.detection_worker.wait(2000):
            QMessageBox.warning(self, "正在关闭", "目标检测线程尚未退出，请稍后再次关闭窗口。")
            event.ignore()
            return
        self.camera.close()
        event.accept()
