"""Estimate optional camera intrinsics from chessboard images."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics from chessboard image files")
    parser.add_argument("--images", required=True, help="Glob pattern, e.g. data/chessboard/*.png")
    parser.add_argument("--rows", type=int, required=True, help="Number of inner chessboard corners per column")
    parser.add_argument("--cols", type=int, required=True, help="Number of inner chessboard corners per row")
    parser.add_argument("--square-mm", type=float, required=True, help="Chessboard square size in mm")
    parser.add_argument("--output", default="data/camera_intrinsic.npz")
    args = parser.parse_args()
    if args.rows < 2 or args.cols < 2:
        parser.error("--rows and --cols must both be at least 2")
    if not np.isfinite(args.square_mm) or args.square_mm <= 0:
        parser.error("--square-mm must be positive and finite")

    paths = [Path(item) for item in sorted(glob.glob(args.images))]
    if not paths:
        raise FileNotFoundError(f"No images matched {args.images!r}")
    pattern = (args.cols, args.rows)
    object_template = np.zeros((args.rows * args.cols, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skip unreadable image: {path}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern)
        if not found:
            print(f"Chessboard not found: {path}")
            continue
        current_size = (gray.shape[1], gray.shape[0])
        if image_size is not None and current_size != image_size:
            raise RuntimeError(
                f"Mixed chessboard image resolutions are not allowed: {path} is {current_size}, "
                f"expected {image_size}"
            )
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_template.copy())
        image_points.append(corners)
        image_size = current_size
        print(f"Accepted: {path}")
    if len(object_points) < 5 or image_size is None:
        raise RuntimeError("Need chessboard detections in at least five images")
    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    if not np.isfinite(rms) or not np.all(np.isfinite(camera_matrix)) or not np.all(np.isfinite(dist_coeffs)):
        raise RuntimeError("Camera calibration produced NaN or infinity")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size),
        reprojection_error=rms,
    )
    print(f"Saved {output}; RMS reprojection error = {rms:.4f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
