"""ADP serial liquid-aspiration controller extracted from the supplied implementation."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from app.config import SuctionConfig
from suction.base import SuctionController


LOGGER = logging.getLogger(__name__)


class ADPCommunicationError(RuntimeError):
    """Raised when the ADP serial port or command transport cannot be confirmed."""


class RealSuctionController(SuctionController):
    """Map one controller suction cycle to one ADP ``n`` aspiration command.

    The supplied protocol contains volume-based aspiration/dispense commands,
    not a continuous vacuum ON/OFF command. Consequently ``on`` sends one
    aspiration command, while ``off`` only closes this software hold state. It
    intentionally does *not* send ``p`` because a safety stop must never eject
    liquid. No physical mid-command cancellation code was present in the
    supplied protocol.
    """

    def __init__(self, config: SuctionConfig) -> None:
        self.config = config
        self._serial: Any | None = None
        self._io_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._active = False
        self._initialized = False
        self._cancel_generation = 0
        self._last_command: str | None = None
        self._last_response = b""

    @staticmethod
    def _serial_module() -> Any:
        try:
            import serial
        except (ImportError, OSError) as exc:
            raise ADPCommunicationError(
                "无法加载 pyserial；请在当前 Python 环境执行 pip install -r requirements.txt"
            ) from exc
        return serial

    @staticmethod
    def _calculate_crc(data: bytes) -> int:
        """Calculate the Modbus-style CRC used by the supplied ASCII frame."""
        value = 0xFFFF
        for byte in data:
            value ^= byte
            for _ in range(8):
                value = (value >> 1) ^ 0xA001 if value & 0x0001 else value >> 1
        return value

    @classmethod
    def _create_command(cls, function_code: str, value: int | None = None) -> str:
        if len(function_code) != 1 or not function_code.isascii():
            raise ValueError("ADP function code must be one ASCII character")
        if value is not None and not 0 <= int(value) <= 0xFFFF:
            raise ValueError(f"ADP command value must be within 0..65535, got {value}")
        data = "" if value is None else f"{int(value):04X}"
        frame = f">01{function_code}{data}"
        return frame + f"{cls._calculate_crc(frame.encode('ascii')):04X}"

    @property
    def last_command(self) -> str | None:
        return self._last_command

    @property
    def last_response(self) -> bytes:
        return bytes(self._last_response)

    def connect(self) -> None:
        """Open and verify ownership of the configured serial port without moving the pipette."""
        with self._io_lock:
            self._ensure_serial_open_locked()

    def _ensure_serial_open_locked(self, max_attempts: int | None = None) -> Any:
        if self._serial is not None and bool(getattr(self._serial, "is_open", False)):
            return self._serial

        serial_module = self._serial_module()
        last_error: Exception | None = None
        attempts = self.config.max_retries if max_attempts is None else max_attempts
        for attempt in range(1, attempts + 1):
            try:
                connection = serial_module.Serial(
                    port=self.config.serial_port,
                    baudrate=self.config.baudrate,
                    timeout=self.config.timeout_s,
                    write_timeout=self.config.timeout_s,
                )
                if not bool(getattr(connection, "is_open", False)):
                    connection.open()
                self._serial = connection
                LOGGER.info("ADP serial port opened: %s", self.config.serial_port)
                return connection
            except Exception as exc:
                last_error = exc
                self._close_serial_locked()
                LOGGER.warning(
                    "ADP serial open attempt %d/%d failed: %s",
                    attempt,
                    attempts,
                    exc,
                )
                if attempt < attempts and self.config.retry_delay_s > 0:
                    time.sleep(self.config.retry_delay_s)
        raise ADPCommunicationError(
            f"打开ADP串口 {self.config.serial_port} 失败，已尝试 {attempts} 次：{last_error}"
        ) from last_error

    def _send_command(self, command: str) -> bytes:
        """Write one complete ADP ASCII frame and optionally require a response."""
        last_error: Exception | None = None
        with self._io_lock:
            for attempt in range(1, self.config.max_retries + 1):
                try:
                    # 命令外层已经负责总重试次数；每轮只做一次重新打开，避免重试次数平方增长。
                    connection = self._ensure_serial_open_locked(max_attempts=1)
                    reset_input = getattr(connection, "reset_input_buffer", None)
                    if callable(reset_input):
                        reset_input()
                    payload = command.encode("ascii")
                    written = connection.write(payload)
                    if written != len(payload):
                        raise IOError(f"serial write accepted {written}/{len(payload)} bytes")
                    flush = getattr(connection, "flush", None)
                    if callable(flush):
                        flush()
                    response = bytes(connection.read(self.config.response_bytes))
                    if self.config.require_response and not response:
                        raise TimeoutError(
                            f"ADP command received no response within {self.config.timeout_s:.3f}s"
                        )
                    self._last_command = command
                    self._last_response = response
                    if response:
                        LOGGER.info("ADP response: %s", response.decode("ascii", errors="replace"))
                    else:
                        LOGGER.warning(
                            "ADP command write completed without response; require_response=false"
                        )
                    return response
                except Exception as exc:
                    last_error = exc
                    self._close_serial_locked()
                    LOGGER.warning(
                        "ADP command attempt %d/%d failed: %s",
                        attempt,
                        self.config.max_retries,
                        exc,
                    )
                    if attempt < self.config.max_retries and self.config.retry_delay_s > 0:
                        time.sleep(self.config.retry_delay_s)
        raise ADPCommunicationError(
            f"ADP命令发送失败，已尝试 {self.config.max_retries} 次：{last_error}"
        ) from last_error

    def on(self) -> None:
        """Optionally initialize/set speed, then command one configured aspiration volume."""
        with self._state_lock:
            if self._active:
                return
            generation = self._cancel_generation

        if self.config.initialize_before_first_absorb and not self._initialized:
            self._send_command(self._create_command("G"))
            self._initialized = True
        if self.config.absorb_speed_ul_s is not None:
            self._send_command(self._create_command("4", self.config.absorb_speed_ul_s))
        self._send_command(self._create_command("n", self.config.absorb_volume_ul))

        with self._state_lock:
            if generation != self._cancel_generation:
                raise ADPCommunicationError(
                    "ADP吸液命令发送期间收到了停止请求；协议未提供中途取消命令，请人工确认设备状态"
                )
            self._active = True

    def off(self) -> None:
        """End the logical hold without sending the protocol's liquid-ejection command."""
        with self._state_lock:
            self._cancel_generation += 1
            self._active = False

    def is_on(self) -> bool:
        """Return cached command state; the supplied protocol has no pressure feedback."""
        with self._state_lock:
            return self._active

    def _close_serial_locked(self) -> None:
        connection = self._serial
        self._serial = None
        if connection is not None:
            try:
                if bool(getattr(connection, "is_open", False)):
                    connection.close()
            except Exception as exc:
                LOGGER.warning("Failed to close ADP serial port: %s", exc)

    def close(self) -> None:
        """Clear cached state and close this controller's serial connection."""
        self.off()
        with self._io_lock:
            self._close_serial_locked()
            self._initialized = False
