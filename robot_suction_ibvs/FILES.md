# 保留文件功能说明

本工程已移除 `tests/`、`pytest.ini`、pytest 依赖及其缓存；以下内容均服务于部署、运行、标定或现场调试。

## 根目录

| 文件/目录 | 功能 |
| --- | --- |
| `gui.py` | PyQt6 桌面界面入口，显示相机画面并启动/停止识别吸取。 |
| `main.py` | 程序入口；组装相机、机械臂、吸盘、视觉和状态机，负责启动及最终安全关闭。 |
| `config.yaml` | 所有可调参数：相机、HSV、尺寸筛选、IBVS、安全和 Z 轴。 |
| `requirements.txt` | 运行所需的 Python 依赖。 |
| `README.md` | 安装、标定、现场运行及真实 SDK 接入说明。 |
| `FILES.md` | 本文件；说明保留文件的职责。 |
| `data/` | 保存相机内参、吸盘参考像素与 IBVS 矩阵等标定产物。 |
| `logs/` | 保存运行日志及可选的控制数据 CSV。 |

## 操作界面

| 文件 | 功能 |
| --- | --- |
| `ui/main_window.py` | 实时相机预览与目标标记、任务状态、开始吸取、停止/急停与运行日志界面。 |
| `ui/detection_worker.py` | 在独立线程中检测最新预览帧，忙时丢弃旧帧，避免目标识别阻塞界面。 |
| `ui/control_worker.py` | 在独立 Qt 线程中运行状态机，避免机械臂 SDK 阻塞界面。 |
| `ui/styles.py` | 集中管理界面的颜色、按钮、状态标签和日志样式。 |

## 核心控制

| 文件 | 功能 |
| --- | --- |
| `app/config.py` | 读取并校验 YAML 配置，提供带类型的配置对象。 |
| `app/state.py` | 定义 INIT、检测、IBVS、下降、吸取、恢复等状态。 |
| `app/controller.py` | 显式状态机；实现检测、锁定、IBVS、吸取和回观察位循环。 |
| `app/logging_utils.py` | 创建会话日志和可选 CSV 控制记录。 |
| `vision/detector.py` | HSV 分割、形态学处理、轮廓提取和像素尺寸筛选。 |
| `vision/models.py` | 检测目标、检测结果和跟踪结果的数据结构。 |
| `vision/tracker.py` | 对锁定目标做最近邻关联，不切换为其他目标。 |
| `vision/visualization.py` | 绘制轮廓、尺寸、吸盘参考点、误差和速度的 OpenCV 叠加层。 |
| `control/ibvs.py` | 根据标定矩阵计算二维 XY 视觉伺服速度，并执行限速/减速/加速度限制。 |
| `control/safety.py` | 所有机器人运动的安全网关：限行程、断流、停止及 Z 轴动作。 |

## 硬件适配

| 文件 | 功能 |
| --- | --- |
| `camera/opencv_camera.py` | USB、视频文件或 RTSP 的 OpenCV 相机实现。 |
| `robot/base.py` | 机械臂统一接口。 |
| `robot/real_robot_template.py` | 真实机械臂 SDK 接入位置；需按品牌 SDK 实现。 |
| `suction/base.py` | 吸盘统一接口。 |
| `suction/real_suction_template.py` | GPIO、串口或 PLC 真空控制接入位置。 |

## 标定与现场工具

| 文件 | 功能 |
| --- | --- |
| `calibration/calibrate_camera.py` | 用棋盘格图像标定可选相机内参和畸变。 |
| `calibration/calibrate_suction_ref.py` | 在人工对准吸盘后保存 `suction_ref.npy`。 |
| `calibration/calibrate_servo_xy.py` | 通过已知 XY 微位移拟合并保存 `servo_A.npy`。 |
| `tools/hsv_tuner.py` | 用滑块现场调 HSV 阈值并保存 JSON。 |
| `tools/inspect_camera.py` | 检查相机设备、分辨率、帧率与断流。 |
| `tools/inspect_detection.py` | 检查 HSV 轮廓、`size_px` 与尺寸筛选结果。 |
