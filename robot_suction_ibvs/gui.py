"""启动 PyQt6 操作界面。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox

from app.config import load_config
from ui.main_window import MainWindow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eye-in-Hand suction operator interface")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().with_name("config.yaml")),
        help="Path to YAML configuration",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("Eye-in-Hand Suction")
    try:
        config = load_config(args.config)
    except Exception as exc:
        QMessageBox.critical(None, "配置加载失败", str(exc))
        return 1
    window = MainWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
