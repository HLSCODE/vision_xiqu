"""OpenCV adapter for USB cameras, video files, and RTSP sources."""

from __future__ import annotations

import platform
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import CameraConfig


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """一帧原图及其在本进程被采集到的时间信息。"""

    bgr: np.ndarray
    sequence: int
    captured_at_monotonic: float


class OpenCVCamera:
    """后台持续采集 OpenCV 图像，并向界面与控制器提供最新 BGR 帧。

    相机只由一个后台线程调用 ``VideoCapture.read()``。界面预览和 IBVS
    控制器调用 ``get_frame()`` 时只复制最新帧，避免两个线程争抢相机数据。
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: cv2.VideoCapture | None = None
        self._capture_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._frame_condition = threading.Condition(self._frame_lock)
        self._consumer_state = threading.local()
        self._latest_frame: CameraFrame | None = None
        self._frame_sequence = 0
        self._actual_resolution = (config.width, config.height)

    def open(self) -> None:
        source: int | str = self.config.video_path or self.config.device_id
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        backend = cv2.CAP_ANY
        if platform.system() == "Linux" and (
            isinstance(source, int)
            or (isinstance(source, str) and source.startswith("/dev/video"))
        ):
            backend = cv2.CAP_V4L2
        elif isinstance(source, int) and platform.system() == "Windows":
            backend = cv2.CAP_DSHOW
        self._capture = cv2.VideoCapture(source, backend)
        if not self._capture.isOpened() and backend != cv2.CAP_ANY:
            self._capture.release()
            self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise RuntimeError(f"Unable to open camera source: {source!r}")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        # V4L2 驱动支持时只保留一个待取帧，减少视觉闭环读到运动前画面的概率。
        # 少数后端会忽略该属性，因此控制器还会校验应用层帧龄。
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        with self._frame_condition:
            self._latest_frame = None
            self._frame_sequence = 0
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="opencv-camera-capture",
            daemon=True,
        )
        self._capture_thread.start()

        # 首次打开时等待一帧，尽早向界面报告设备、权限或数据流错误。
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            actual_resolution: tuple[int, int] | None = None
            with self._frame_lock:
                if self._latest_frame is not None:
                    height, width = self._latest_frame.bgr.shape[:2]
                    actual_resolution = (int(width), int(height))
            if actual_resolution is not None:
                self._actual_resolution = actual_resolution
                requested = (int(self.config.width), int(self.config.height))
                if actual_resolution != requested:
                    self.close()
                    raise RuntimeError(
                        f"相机实际分辨率 {actual_resolution[0]}x{actual_resolution[1]} 与配置 "
                        f"{requested[0]}x{requested[1]} 不一致。像素标定不能缩放复用，请修改配置或相机格式后重新标定"
                    )
                return
            time.sleep(0.01)
        self.close()
        raise RuntimeError(f"Camera opened but no frame arrived from source: {source!r}")

    def _capture_loop(self) -> None:
        """相机采集线程：始终保存最新帧，不积压历史帧。"""
        while not self._stop_event.is_set():
            capture = self._capture
            if capture is None:
                break
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.02)
                continue
            with self._frame_condition:
                self._frame_sequence += 1
                self._latest_frame = CameraFrame(
                    bgr=frame,
                    sequence=self._frame_sequence,
                    captured_at_monotonic=time.monotonic(),
                )
                self._frame_condition.notify_all()

    def close(self) -> None:
        self._stop_event.set()
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()
        thread = self._capture_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._capture_thread = None
        with self._frame_condition:
            self._latest_frame = None
            self._frame_condition.notify_all()

    def get_frame_packet(self) -> CameraFrame | None:
        """等待本调用线程尚未读取的新帧，并返回带采集时间的原图副本。"""
        if self._capture is None:
            raise RuntimeError("Camera is not open")
        previous_sequence = getattr(self._consumer_state, "last_sequence", -1)
        with self._frame_condition:
            self._frame_condition.wait_for(
                lambda: self._capture is None
                or (
                    self._latest_frame is not None
                    and self._latest_frame.sequence != previous_sequence
                ),
                timeout=1.0,
            )
            if (
                self._capture is None
                or self._latest_frame is None
                or self._latest_frame.sequence == previous_sequence
            ):
                return None
            packet = self._latest_frame
            self._consumer_state.last_sequence = packet.sequence
            return CameraFrame(
                bgr=packet.bgr.copy(),
                sequence=packet.sequence,
                captured_at_monotonic=packet.captured_at_monotonic,
            )

    def get_frame(self) -> np.ndarray | None:
        """兼容旧调用：返回本调用线程尚未读取的新原始分辨率图像。"""
        packet = self.get_frame_packet()
        return None if packet is None else packet.bgr

    def get_preview_frame(self, max_width: int = 1920, max_height: int = 1080) -> np.ndarray | None:
        """返回适合界面显示的缩放帧，避免反复复制 4024×3036 原图。"""
        if self._capture is None:
            raise RuntimeError("Camera is not open")
        with self._frame_lock:
            packet = self._latest_frame
        if packet is None:
            return None
        frame = packet.bgr
        height, width = frame.shape[:2]
        scale = min(max_width / width, max_height / height, 1.0)
        if scale >= 1.0:
            return frame.copy()
        return cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def get_resolution(self) -> tuple[int, int]:
        return self._actual_resolution
