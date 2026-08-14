# Eye-in-Hand 视觉伺服小目标自动吸取系统

本工程实现固定观察高度的小目标自动吸取流程：HSV 黑色目标分割、像素尺寸筛选、单目标最近邻锁定、可见区二维 IBVS 预对准、受限终末 XY 移动、Z 轴下降、ADP 定量吸液、抬升和返回观察位。

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

   `calibrate_servo_xy.py` 运行期间画面中必须始终只有一个尺寸合格且静止的目标；
   工具会检查采样抖动、机械臂微移后回位误差、拟合秩、条件数与拟合 RMS，不合格时不会保存矩阵。
   每次运动指令完成后，脚本先等待机械臂报告停止；随后默认用 1 秒持续读取并丢弃振动、
   相机缓存和自动曝光过渡帧，最后才在最长 10 秒内寻找连续稳定的 10 帧窗口并计算中心。
   曝光或结构稳定较慢时可增加：

   ```bash
   python calibration/calibrate_servo_xy.py --config config.yaml --frames 10 \
     --settle-time-s 2 --sample-timeout-s 20
   ```

   不建议直接把 `--max-jitter-px` 从 3 调到几十或上百；这种波动通常表示自动曝光尚未稳定、
   黑色轮廓发生粘连/分裂、画面中存在多个合格目标，或相机/目标/机械臂仍在移动。

   每次微移后脚本都以与开始时完全相同的 `config.robot.observe_pose` 绝对回位；该指令采用与手动
   输入原位姿相同的关节规划方式，而不是根据中间的位姿反馈再计算一条相对回位指令。矩阵拟合仍采用
   每次移动前后的真实机械臂反馈位移，不假定指令中的 `1/2 mm` 被完全执行。回到观察位并完成稳定帧
   采样后，原始图像中心差必须不超过 `--max-return-error-px`（默认 `3 px`）；超限时应检查
   `config.robot.observe_pose`、当前工作坐标系、目标是否移动，或 HSV 是否切换到另一轮廓，而不是放宽阈值。

   预对准标定画面中也只保留同一个尺寸合格目标：吸管对准目标且目标中心可检测时保持静止，
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

`ibvs.prealign_offset_confirmed` 同样默认为 `false`。运行预对准标定并确认目标在预对准点完整可见后，使用 `--apply-config --confirm-target-visible` 写入偏移并打开该开关。

所有运动命令由 `control/safety.py` 统一限制，覆盖最大 XY 行程、终末 XY 单次上限、相机断流、对准超时、连续误差增长与安全停止。首次真实运行请先把 `pick_z_mm` 设置在不会接触目标的测试高度，验证预对准和终末 XY 方向后，再逐步降低到真实吸取高度，并保留硬件急停。

详细的文件职责请参阅 [FILES.md](FILES.md)。
