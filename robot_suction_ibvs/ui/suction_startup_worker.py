"""One-time ADP startup preparation without blocking the Qt event loop."""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from suction.adp_suction import RealSuctionController


class SuctionStartupWorker(QThread):
    """Open the ADP serial port and issue the one-time startup commands.

    The worker owns no hardware state. It uses the long-lived controller created
    by the main window, which is later handed to every control-worker run.
    """

    ready = pyqtSignal()
    failed = pyqtSignal(str)
    message = pyqtSignal(str)

    def __init__(self, suction: RealSuctionController, parent=None) -> None:
        super().__init__(parent)
        self._suction = suction

    def run(self) -> None:
        try:
            self.message.emit("正在连接 ADP 吸液枪……")
            self._suction.connect()
            if self._suction.config.initialize_on_startup:
                self.message.emit("正在执行一次 ADP 枪头脱落初始化……")
            self._suction.initialize_for_application()
            self.ready.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
