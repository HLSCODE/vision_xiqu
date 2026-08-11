# Eye-in-Hand 视觉伺服小目标自动吸取系统

本工程实现固定观察高度的小目标自动吸取流程：HSV 黑色目标分割、像素尺寸筛选、单目标最近邻锁定、可见区二维 IBVS 预对准、受限终末 XY 移动、Z 轴下降、吸盘开启、抬升和返回观察位。

不包含 YOLO、RGB-D、深度估计、PBVS、完整手眼标定 AX=XB、世界坐标转换或复杂多目标跟踪。

## 安装

```bash
cd robot_suction_ibvs
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ubuntu 上若使用 USB 相机，可先确认实际设备节点：

```bash
sudo apt install v4l-utils
v4l2-ctl --list-devices
```

在 `config.yaml` 中设置 `camera.device_id`（或 `camera.video_path`），并确认睿尔曼控制器 IP、观察位姿与吸盘真实接口已经配置。

## 工作原理

在 `GLOBAL_DETECT` 状态，图像经 HSV 阈值分割、开闭运算和轮廓提取。每个轮廓计算质心 `[u,v]`、面积、旋转矩形宽高和 `size_px = max(width_px, height_px)`。仅 `size_px < vision.size_threshold_px` 的对象能被选作吸取目标。

进入 `ALIGN_IBVS` 后，不再重新做尺寸筛选或重新选择对象；只在当前帧中关联距上一帧锁定目标最近的轮廓。IBVS 将目标对准到吸管遮挡区外的预对准点：

```text
prealign_ref = suction_ref + prealign_offset_px
e_px = [u-u_prealign, v-v_prealign]^T
v_xy_mm_s = -gain * inv(A_px_per_mm) * e_px
```

其中 `A_px_per_mm` 在固定观察高度标定。目标在预对准点连续满足 `pixel_tolerance` 与 `stable_frames` 后，系统用最后一帧计算一次终末位置移动：

```text
final_dXY_mm = inv(A_px_per_mm) * (suction_ref - current_pixel)
```

该位移必须小于 `final_approach_max_mm`，并由机械臂相对位置接口执行。运动途中即使吸管遮住目标也不再依赖视觉；规划运动完成后 XY 保持停止，只执行教示的 Z 轴下降和吸取动作。目标在预对准阶段丢失仍会进入安全恢复，不会被当成对准成功。

## 标定与现场调试

1. 调 HSV：

   ```bash
   python tools/hsv_tuner.py --config config.yaml
   python tools/inspect_detection.py --config config.yaml
   ```

2. 可选相机内参和畸变标定：

   ```bash
   python calibration/calibrate_camera.py \
     --images 'data/chessboard/*.png' --rows 6 --cols 9 --square-mm 20
   ```

3. 在固定观察高度用已知 XY 微位移拟合二维视觉映射矩阵：

   ```bash
   python calibration/calibrate_servo_xy.py --config config.yaml --frames 10
   ```

4. 运行双位置图像标定，同时生成吸管参考点和预对准偏移：

   ```bash
   python calibration/calibrate_prealign.py --config config.yaml --frames 20
   ```

   标定画面中只保留同一个尺寸合格目标：吸管对准目标且目标中心可检测时保持静止，
   等待采满连续帧后按 `A`；然后保持 Z、姿态和目标不变，只移动 XY 到目标完全无遮挡的
   预对准位置，再次等待采满连续帧后按 `P`。

   脚本直接计算：

   ```text
   prealign_offset_px = 预对准目标中心 - 对准目标中心
   运行时终末XY       = inv(servo_A) @ (-prealign_offset_px)
   ```

   两处采样都通过稳定性、矩阵和安全距离检查后，脚本将对准位目标中心保存为
   `data/suction_ref.npy`，并输出预对准偏移和终末 XY。确认预对准位置目标完整可见后，可以显式写入配置：

   ```bash
   python calibration/calibrate_prealign.py --config config.yaml --frames 20 \
     --apply-config --confirm-target-visible
   ```

   工具不会连接或移动机械臂，只读取相机。两次采样必须保持相机分辨率、Z、姿态和目标位置不变。如果对准时吸管遮挡目标，可以按你的方法拆下吸管完成两处采样，但拆装不能改变相机位置或末端姿态；配置应用后还必须重新装好吸管，确认预对准位置确实无遮挡。

真实启动要求以下两个文件存在且格式正确；缺少时程序会在运动前拒绝启动：

- `data/suction_ref.npy`
- `data/servo_A.npy`

## 启动

完成标定、确认速度和 Z 高度安全、实现吸盘真实硬件适配器后：

```bash
python gui.py
```

桌面界面使用 PyQt6 编写，会自动连接 OpenCV 相机并显示实时画面。检测到的目标会显示轮廓、中心十字、编号和像素尺寸：绿色表示尺寸符合吸取阈值，橙色表示目标超过尺寸阈值。确认相机画面、机械臂安全区域和吸盘状态后，点击“开始识别吸取”运行完整状态机；“停止 / 急停”会停止机械臂并关闭吸盘。

## 接入真实硬件

机械臂适配器使用睿尔曼官方 Python API2；吸盘控制仍需根据实际 GPIO、串口或 PLC 接口实现：

- `robot/realman_robot.py`：连接 `config.yaml` 中的控制器 IP，完成点位、XY 速度透传、相对 XY、Z 运动和停止；工程单位为 mm / mm/s，适配层转换为 SDK 的 m / m/s。
- `camera/opencv_camera.py`：通过 OpenCV 读取 USB 相机、视频文件或 RTSP 视频流。
- `suction/real_suction_template.py`：GPIO、串口或 PLC 的真空开关控制。

首次接入时，`robot.observe_pose_confirmed` 和 `robot.z_motion_confirmed` 默认为 `false`。只有在示教器中核对观察位及全部 Z 高度后才可改为 `true`；否则适配器会拒绝相关运动。

所有运动命令由 `control/safety.py` 统一限制，覆盖最大 XY 行程、终末 XY 单次上限、相机断流、对准超时、连续误差增长与安全停止。首次真实运行请先把 `pick_z_mm` 设置在不会接触目标的测试高度，验证预对准和终末 XY 方向后，再逐步降低到真实吸取高度，并保留硬件急停。

详细的文件职责请参阅 [FILES.md](FILES.md)。
