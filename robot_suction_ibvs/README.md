# Eye-in-Hand 视觉伺服小目标自动吸取系统

本工程实现固定观察高度的小目标自动吸取流程：HSV 黑色目标分割、像素尺寸筛选、单目标最近邻锁定、二维 IBVS 直接对准吸管轴线、Z 轴下降、ADP 定量吸液、抬升和返回观察位。

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

确认 ADP 串口及当前用户权限：

```bash
ls -l /dev/ttyUSB*
groups
# 若当前用户不在 dialout 组：
sudo usermod -aG dialout "$USER"
```

修改用户组后需要注销并重新登录，再把实际设备节点填入 `suction.serial_port`。

在 `config.yaml` 中设置 `camera.device_id`（或 `camera.video_path`），并确认睿尔曼控制器 IP、观察位姿与 `suction.serial_port` 已经配置。Ubuntu 下还需确认运行用户有串口读写权限。

## 工作原理

在 `GLOBAL_DETECT` 状态，图像经 HSV 阈值分割、开闭运算和轮廓提取。每个轮廓计算质心 `[u,v]`、面积、旋转矩形宽高和 `size_px = max(width_px, height_px)`。仅 `size_px < vision.size_threshold_px` 的对象能被选作吸取目标。

进入 `ALIGN_IBVS` 后，不再重新做尺寸筛选或重新选择对象；只在当前帧中关联距上一帧锁定目标最近的轮廓。IBVS 直接将目标中心对准吸管轴线参考点：

```text
e_px = [u-u_ref, v-v_ref]^T
v_xy_mm_s = -gain * inv(A_px_per_mm) * e_px
```

其中 `A_px_per_mm` 在固定观察高度标定，`suction_ref` 是吸管轴线正对目标且目标仍可见时采集的目标中心。目标连续满足 `pixel_tolerance` 与 `stable_frames` 后，系统停止 XY 并直接执行教示的 Z 轴下降和吸取动作。

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

4. 将吸管轴线正对一个仍可见的静止目标，运行单点标定并按 `A` 保存吸管参考像素：

   ```bash
   python calibration/calibrate_suction_ref.py --config config.yaml --frames 20
   ```

   `calibrate_servo_xy.py` 运行期间画面中必须始终只有一个尺寸合格且静止的目标；
   工具会检查采样抖动、拟合秩、条件数与拟合 RMS，不合格时不会保存矩阵。
   每次运动指令完成后，脚本先等待机械臂报告停止；随后默认用 1 秒持续读取并丢弃振动、
   相机缓存和自动曝光过渡帧，最后才在最长 10 秒内寻找连续稳定的 10 帧窗口并计算中心。
   曝光或结构稳定较慢时可增加：

   ```bash
   python calibration/calibrate_servo_xy.py --config config.yaml --frames 10 \
     --settle-time-s 2 --sample-timeout-s 20
   ```

   不建议直接把 `--max-jitter-px` 从 3 调到几十或上百；这种波动通常表示自动曝光尚未稳定、
   黑色轮廓发生粘连/分裂、画面中存在多个合格目标，或相机/目标/机械臂仍在移动。

   `calibrate_servo_xy.py` 是一个独立的八次有限差分标定：`X-`、`X+`、`Y-`、`Y+` 四个方向各移动
   两次。每一次试验都严格执行：固定 `config.robot.observe_pose` → 等待稳定并采样中心 → 相对移动
   `±2 mm` → 等待稳定并采样中心 → 回到同一个固定观察位 → 等待稳定并再次采样中心。
   最后的回位中心只输出日志，便于人工核对，不参与补偿或重试；矩阵只由八组“移动前后机械臂实际
   XY 反馈位移”与“移动前后图像中心位移”拟合。可按需要调整步长或重复次数：

   ```bash
   python calibration/calibrate_servo_xy.py --config config.yaml --step-mm 2 --repeats 2
   ```

   吸管参考点标定工具不会连接或移动机械臂，只读取相机。吸管轴线已正对目标、目标中心可见且稳定时按 `A`；它会将单次稳定采样的中心保存为 `data/suction_ref.npy`。

真实启动要求以下两个数组及其同名 `.json` 标定上下文存在且格式正确；缺少时程序会在运动前拒绝启动：

- `data/suction_ref.npy`
- `data/servo_A.npy`

上下文记录并校验相机分辨率、畸变校正开关与内参指纹、观察位姿、机械臂工作坐标系，
以及 `suction_ref` 对应的 `servo_A` 文件指纹。上述任意条件改变后必须重新标定，不能按比例复用旧像素坐标。

## 启动

完成标定，并确认速度、Z 高度和 ADP 串口参数安全后：

```bash
python gui.py
```

桌面界面使用 PyQt6 编写，会自动连接 OpenCV 相机并显示实时画面。检测到的目标会显示轮廓、编号和原始采集图像坐标 `(u,v)`，不绘制中心十字：绿色表示尺寸符合吸取阈值，橙色表示目标超过尺寸阈值。即使界面预览经过压缩，标签中的坐标仍属于控制器和标定使用的原图。确认相机画面、机械臂安全区域和吸液枪状态后，点击“开始识别吸取”运行完整状态机。

## 接入真实硬件

机械臂适配器使用睿尔曼官方 Python API2；吸液枪使用附件提供的 ADP ASCII/CRC 串口协议：

- `robot/realman_robot.py`：连接 `config.yaml` 中的控制器 IP，完成点位、XY 速度透传、相对 XY、Z 运动和停止；工程单位为 mm / mm/s，适配层转换为 SDK 的 m / m/s。
- `camera/opencv_camera.py`：通过 OpenCV 读取 USB 相机、视频文件或 RTSP 视频流。
- `suction/adp_suction.py`：连接配置串口，可选发送 `G` 初始化和 `4` 吸液速度命令，每次吸取发送一次 `n + 体积` 命令。

ADP 协议注意事项：

- `suction.absorb_volume_ul` 决定每个目标的吸液体积；`absorb_speed_ul_s: null` 表示沿用设备当前速度。
- 若设置了吸液速度，配置加载会要求 `hold_time_s >= 体积 / 速度`，防止动作未完成就抬升。
- 附件没有提供“中途停止吸液”命令，因此程序的 `off()` 只结束本轮软件状态，绝不会在急停时发送 `p` 吐液。
- `is_on()` 是已成功写入吸液命令的缓存状态，不是压力或液体检测反馈。
- `require_response` 默认保持附件原逻辑为 `false`；确认设备确实返回应答后，建议改为 `true`。
- 当前流程不自动吐液。达到吸液枪容量前，需要另行安排人工排液或后续安全排液流程。

首次接入时，`robot.observe_pose_confirmed` 和 `robot.z_motion_confirmed` 默认为 `false`。只有在示教器中核对观察位及全部 Z 高度后才可改为 `true`；否则适配器会拒绝相关运动。

所有运动命令由 `control/safety.py` 统一限制，覆盖最大 XY 行程、相机断流、对准超时、连续误差增长与安全停止。首次真实运行请先把 `pick_z_mm` 设置在不会接触目标的测试高度，验证直接 IBVS 方向后，再逐步降低到真实吸取高度，并保留硬件急停。

详细的文件职责请参阅 [FILES.md](FILES.md)。
