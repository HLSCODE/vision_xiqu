"""CLI entry point for the Eye-in-Hand small-object suction system."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from app.config import AppConfig, load_config
from app.controller import SuctionRobotController
from app.logging_utils import SessionLogger
from camera.opencv_camera import OpenCVCamera
from control.ibvs import IBVSController
from control.safety import SafetyManager
from robot.real_robot_template import RealRobot
from suction.real_suction_template import RealSuctionController
from vision.detector import HSVObjectDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eye-in-Hand HSV/IBVS suction controller")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().with_name("config.yaml")),
        help="Path to YAML configuration",
    )
    parser.add_argument("--no-gui", action="store_true", help="Disable OpenCV debug windows")
    parser.add_argument("--max-steps", type=int, default=None, help="Emergency-stop after this many control cycles")
    return parser.parse_args()


def build_adapters(config: AppConfig) -> tuple[OpenCVCamera, RealRobot, RealSuctionController]:
    """Create real hardware adapters while keeping vendor SDK code isolated."""
    # 如接入不同品牌硬件，只替换这三个适配器的实现；视觉与控制层无需修改。
    return OpenCVCamera(config.camera), RealRobot(), RealSuctionController()


def load_runtime_calibration(config: AppConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load mandatory real-hardware calibration artifacts before any robot motion."""
    # A 描述固定观察高度下“机械臂 XY 位移 -> 图像像素位移”的局部线性关系。
    servo_path = config.path(config.ibvs.servo_A_path)
    # suction_ref 是吸盘轴线正好对准目标时，目标在图像中的像素坐标，不一定在图像中心。
    ref_path = config.path(config.suction.suction_ref_path)
    missing = [str(path) for path in (servo_path, ref_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Real hardware motion is blocked: missing calibration file(s): " + ", ".join(missing)
            + ". Run calibration/calibrate_suction_ref.py and calibration/calibrate_servo_xy.py first."
        )
    matrix = np.load(servo_path)
    reference = np.load(ref_path)
    if matrix.shape != (2, 2) or reference.shape != (2,):
        raise ValueError("Invalid calibration shapes: servo_A must be (2,2), suction_ref must be (2,)")
    return matrix.astype(np.float64), reference.astype(np.float64)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.no_gui:
        # 命令行覆盖只影响本次进程，不会改写用户的 config.yaml。
        config = replace(config, system=replace(config.system, show_debug=False, show_mask=False))
    session = SessionLogger(config.path("logs"), config.system.save_csv)
    camera: OpenCVCamera | None = None
    robot: RealRobot | None = None
    suction: RealSuctionController | None = None
    try:
        # 先加载标定文件，再打开任何运动能力，避免未标定机械臂进入控制循环。
        camera, robot, suction = build_adapters(config)
        servo_A, suction_ref = load_runtime_calibration(config)
        detector = HSVObjectDetector(
            config.vision,
            intrinsic_path=config.path(config.camera.intrinsic_path),
            enable_undistort=config.camera.enable_undistort,
        )
        ibvs = IBVSController(servo_A, config.ibvs)
        session.logger.info("servo_A condition number: %.3f", ibvs.condition_number)
        camera.open()
        robot.connect()
        controller = SuctionRobotController(
            config, camera, detector, SafetyManager(robot, config.safety), suction, ibvs, suction_ref, session
        )
        state = controller.run(max_steps=args.max_steps)
        session.logger.info("Controller finished in %s; picks=%d", state.name, controller.pick_count)
        return 0 if state.name == "FINISHED" else 1
    except KeyboardInterrupt:
        session.logger.warning("Keyboard interrupt received")
        return 130
    except Exception as exc:
        session.logger.exception("Startup or run failure: %s", exc)
        return 1
    finally:
        # 无论正常结束、SDK 异常或 Ctrl+C，都按“停机器人、关吸盘、关相机”顺序清理。
        if robot is not None:
            robot.stop_all()
            robot.disconnect()
        if suction is not None:
            suction.off()
            suction.close()
        if camera is not None:
            camera.close()
        session.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    sys.exit(main())
