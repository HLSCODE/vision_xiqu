"""Structured session logging to text and optional CSV."""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "timestamp", "state", "u", "v", "u_ref", "v_ref", "error_u", "error_v",
    "vx", "vy", "size_px", "tracking_distance",
]


class SessionLogger:
    """Own a session-specific Python logger and optional IBVS CSV recorder."""

    def __init__(self, log_dir: Path, save_csv: bool) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = logging.getLogger(f"robot_suction_ibvs.{stamp}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler = logging.FileHandler(log_dir / f"session_{stamp}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        self._csv_file = None
        self._csv_writer: csv.DictWriter[str] | None = None
        if save_csv:
            self._csv_file = (log_dir / f"session_{stamp}.csv").open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS)
            self._csv_writer.writeheader()

    def record(self, **values: Any) -> None:
        """Append one control-cycle record while allowing absent fields."""
        if self._csv_writer is None:
            return
        row = {field: values.get(field, "") for field in CSV_FIELDS}
        row["timestamp"] = values.get("timestamp", datetime.now().isoformat(timespec="milliseconds"))
        self._csv_writer.writerow(row)
        self._csv_file.flush()  # type: ignore[union-attr]

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
