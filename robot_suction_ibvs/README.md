# Eye-in-Hand 视觉伺服小目标自动吸取系统

本工程实现第一版固定观察高度的小目标自动吸取流程：HSV 橙黄色分割、像素尺寸筛选、单目标最近邻锁定、二维 IBVS XY 对准、Z 轴下降、吸盘开启、抬升和返回观察位。

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

在 `config.yaml` 中设置 `camera.device_id`（或 `camera.video_path`），并确认机械臂与吸盘的真实 SDK 适配器已经实现。

## 工作原理

在 `GLOBAL_DETECT` 状态，图像经 HSV 阈值分割、开闭运算和轮廓提取。每个轮廓计算质心 `[u,v]`、面积、旋转矩形宽高和 `size_px = max(width_px, height_px)`。仅 `size_px < vision.size_threshold_px` 的对象能被选作吸取目标。

进入 `ALIGN_IBVS` 后，不再重新做尺寸筛选或按吸盘参考点重新选择对象；只在当前帧中关联距上一帧锁定目标最近的轮廓。控制律为：

```text
e_px = [u-u_ref, v-v_ref]^T
v_xy_mm_s = -gain * inv(A_px_per_mm) * e_px
```

其中 `A_px_per_mm` 在固定观察高度标定。对准连续满足 `pixel_tolerance` 与 `stable_frames` 后，XY 停止；随后仅执行教示的 Z 轴下降和吸取动作。

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

3. 在吸盘与一个目标人工精确对准后，记录吸盘参考像素：

   ```bash
   python calibration/calibrate_suction_ref.py --config config.yaml --frames 20
   ```

4. 在固定观察高度用已知 XY 微位移拟合二维视觉映射矩阵：

   ```bash
   python calibration/calibrate_servo_xy.py --config config.yaml --frames 10
   ```

真实启动要求以下两个文件存在且格式正确；缺少时程序会在运动前拒绝启动：

- `data/suction_ref.npy`
- `data/servo_A.npy`

## 启动

完成标定、确认速度和 Z 高度安全、实现真实硬件适配器后：

```bash
python gui.py
```

桌面界面使用 PyQt6 编写，会自动连接 OpenCV 相机并显示实时画面。检测到的目标会显示轮廓、中心十字、编号和像素尺寸：绿色表示尺寸符合吸取阈值，橙色表示目标超过尺寸阈值。确认相机画面、机械臂安全区域和吸盘状态后，点击“开始识别吸取”运行完整状态机；“停止 / 急停”会停止机械臂并关闭吸盘。

也可不使用桌面界面，直接从命令行运行控制器：

```bash
python main.py --no-gui
```

`--max-steps N` 可限制本次控制循环步数。

## 接入真实硬件

本工程没有虚构任何硬件 SDK 调用。只需在以下模板中使用厂家已验证的 API 实现接口，无需修改视觉、IBVS 或状态机代码：

- `robot/real_robot_template.py`：机械臂连接、点位、XY 速度、相对 XY、Z 运动和停止；单位必须为 mm / mm/s。
- `camera/opencv_camera.py`：通过 OpenCV 读取 USB 相机、视频文件或 RTSP 视频流。
- `suction/real_suction_template.py`：GPIO、串口或 PLC 的真空开关控制。

所有运动命令由 `control/safety.py` 统一限制，覆盖最大 XY 行程、相机断流、对准超时、连续误差增长与安全停止。首次真实运行请使用低速度、单目标、安全 Z 高度，并保留硬件急停。

详细的文件职责请参阅 [FILES.md](FILES.md)。
