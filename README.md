# Ascend YOLOv8 Sample

## 项目介绍

本项目是基于华为昇腾 Ascend 平台的 YOLOv8 C++ 推理示例，主要用于在 Atlas/Ascend 设备上运行图片检测、单路视频流检测、多路 RTSP 检测和性能测试。

代码基于 Ascend Samples 中的 YOLO 示例进行改造，适配 YOLOv8 输出格式。YOLOv8 常见检测模型输出为 `[1, 84, 8400]`，和 YOLOv7 的 `[1, 25200, 85]` 不同，因此后处理逻辑做了对应调整。

当前项目重点支持：

- 图片目录批量推理
- 单路 RTSP 或本地视频文件推理
- 多路 RTSP 线程模式推理
- 多路 RTSP 多进程模式推理
- 模型推理耗时统计，包括拉流解码、预处理、推理、D2H、后处理、总耗时和 FPS
- YOLOv8 Detect/Seg/Cls/Obb/Pose 示例编译，其中部分目标依赖 `yaml-cpp`

## 目录结构

```text
ascend-yolov8-sample-master/
├── common/                 # AclLite 公共封装和 DVPP/视频处理工具
├── data/                   # 测试图片目录，仓库中可不提交大数据
├── imgs/                   # README 或结果示意图
├── src/
│   ├── CMakeLists.txt      # 编译入口
│   ├── model/              # OM/ONNX 模型目录，模型文件不提交
│   ├── out/                # 编译后的可执行文件输出目录
│   ├── scripts/            # 构建/运行脚本
│   ├── bench_rtsp_models.sh
│   ├── sampleYOLOV8.cpp
│   ├── sampleYOLOV8_npu.cpp
│   ├── sampleYOLOV8_npu-rtsp.cpp
│   ├── sampleYOLOV8_npu-rtsp-thread.cpp
│   └── sampleYOLOV8_npu-rtsp-process.cpp
└── README.md
```

## 环境依赖

运行环境需要提前准备：

- Ascend 设备，例如 Atlas 500 Pro 或其他 Ascend 310/310P/310B 系列设备
- Ubuntu/Linux 环境
- 已安装固件、驱动、CANN/Ascend Toolkit
- 已安装或编译可用的 `libacllite.so`
- `gcc`、`g++`、`cmake`、`make`
- OpenCV
- FFmpeg 相关库，项目会从 Ascend Toolkit 的 thirdpart 路径查找
- `yaml-cpp`，可选；缺少时 `sampleYOLOV8Seg/Cls/Obb/Pose` 会跳过

可先确认 NPU 状态：

```bash
npu-smi info
```

如果运行时报动态库找不到，可以补充：

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/thirdpart/lib:$LD_LIBRARY_PATH
```

如果你的环境使用的是 `/usr/local/Ascend/thirdpart/aarch64/lib`，也可以按实际路径追加到 `LD_LIBRARY_PATH`。

## 模型准备

模型文件放在：

```text
src/model/
```

常用模型示例：

```text
person_det.om
person_det_static.om
person_det_640.om
person_det_640_static.om
1280_s.om
1280_s_static.om
yolov8s.om
yolov8s_static.om
```

模型文件通常较大，仓库已通过 `.gitignore` 忽略 `*.om`、`*.onnx`、`*.zip`，不会上传模型和压缩包。

ONNX 转 OM 示例：

```bash
atc \
  --model=yolov8s.onnx \
  --framework=5 \
  --output=yolov8s \
  --input_shape="images:1,3,640,640" \
  --soc_version=Ascend310B1 \
  --insert_op_conf=aipp.cfg
```

1280 输入模型示例：

```bash
atc \
  --model=1280_s.onnx \
  --framework=5 \
  --output=1280_s \
  --input_shape="images:1,3,1280,1280" \
  --soc_version=Ascend310B1 \
  --insert_op_conf=aipp_1280.cfg
```

`--soc_version` 需要和实际设备匹配，可通过 `npu-smi info` 查看。

## 编译

进入源码目录：

```bash
cd ascend-yolov8-sample-master/src
```

推荐使用独立构建目录：

```bash
mkdir -p build/intermediates/host
cd build/intermediates/host
cmake ../../../ -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

编译完成后，可执行文件输出到：

```text
src/out/
```

也可以使用脚本编译：

```bash
cd ascend-yolov8-sample-master/src
bash scripts/sample_build.sh
```

脚本默认会检查 `src/model/yolov8s.om` 和 `src/model/yolov8s_static.om` 是否存在。

## 程序说明

主要可执行文件如下：

| 程序 | 说明 |
|---|---|
| `sampleYOLOV8` | 图片目录推理，基础版本 |
| `sampleYOLOV8_npu` | 图片目录推理，支持设备号、绘图开关、耗时统计 |
| `sampleYOLOV8_npu_rtsp` | 单路 RTSP 或本地视频文件推理 |
| `sampleYOLOV8_npu_rtsp_thread` | 多路流线程模式推理 |
| `sampleYOLOV8_npu_rtsp_process` | 多路流多进程模式推理 |
| `sampleYOLOV8Seg` | YOLOv8 分割示例，依赖 `yaml-cpp` |
| `sampleYOLOV8Cls` | YOLOv8 分类示例，依赖 `yaml-cpp` |
| `sampleYOLOV8Obb` | YOLOv8 OBB 示例，依赖 `yaml-cpp` |
| `sampleYOLOV8Pose` | YOLOv8 Pose 示例，依赖 `yaml-cpp` |

## 图片推理

进入输出目录：

```bash
cd ascend-yolov8-sample-master/src/out
```

命令格式：

```bash
./sampleYOLOV8_npu [model_path] [model_width] [model_height] [image_dir] [draw_result] [max_log_det] [device_id]
```

参数说明：

| 参数 | 说明 |
|---|---|
| `model_path` | OM 模型路径 |
| `model_width` | 模型输入宽度，需为 32 的倍数 |
| `model_height` | 模型输入高度，需为 32 的倍数 |
| `image_dir` | 图片目录 |
| `draw_result` | 是否绘制结果，`1` 开启，`0` 关闭 |
| `max_log_det` | 每张图最多打印多少个检测结果 |
| `device_id` | NPU 设备号 |

示例：

```bash
./sampleYOLOV8_npu ../model/yolov8s.om 640 640 ../data 1 3 0
```

关闭绘图示例：

```bash
./sampleYOLOV8_npu ../model/yolov8s.om 640 640 ../data 0 3 0
```

绘图结果默认保存到：

```text
src/out/results/
```

## 单路 RTSP 或视频推理

命令格式：

```bash
./sampleYOLOV8_npu_rtsp "<rtsp_url_or_video_file>" [model_path] [model_width] [model_height] [frame_limit] [max_log_det] [device_id] [tcp|udp] [stats_interval_sec]
```

参数说明：

| 参数 | 说明 |
|---|---|
| `rtsp_url_or_video_file` | 单路 RTSP 地址或本地视频文件路径 |
| `model_path` | OM 模型路径 |
| `model_width` | 模型输入宽度 |
| `model_height` | 模型输入高度 |
| `frame_limit` | 处理帧数，`0` 表示持续运行 |
| `max_log_det` | 每帧最多打印多少个检测结果 |
| `device_id` | NPU 设备号 |
| `tcp|udp` | RTSP 传输方式 |
| `stats_interval_sec` | 周期统计日志间隔，单位秒 |

示例：

```bash
./sampleYOLOV8_npu_rtsp "<rtsp_url>" ../model/yolov8s_static.om 640 640 10000 3 0 tcp 5
```

本地视频文件示例：

```bash
./sampleYOLOV8_npu_rtsp "../data/test.mp4" ../model/yolov8s_static.om 640 640 10000 3 0 tcp 5
```

## 多路 RTSP 线程模式

线程模式适合快速验证多路流处理能力。多路地址用英文逗号连接。

命令格式：

```bash
./sampleYOLOV8_npu_rtsp_thread "<rtsp_url_1>,<rtsp_url_2>" [model_path] [model_width] [model_height] [frame_limit] [max_log_det] [device_id] [tcp|udp] [stats_interval_sec]
```

示例：

```bash
./sampleYOLOV8_npu_rtsp_thread "<rtsp_url_1>,<rtsp_url_2>" ../model/yolov8s_static.om 640 640 0 3 0 tcp 5
```

限制帧数示例：

```bash
./sampleYOLOV8_npu_rtsp_thread "<rtsp_url_1>,<rtsp_url_2>" ../model/yolov8s_static.om 640 640 10000 3 0 tcp 5
```

## 多路 RTSP 多进程模式

多进程模式会为每路流启动独立子进程，适合对比线程模式和进程模式的稳定性、吞吐和资源占用。

命令格式：

```bash
./sampleYOLOV8_npu_rtsp_process "<rtsp_url_1>,<rtsp_url_2>" [model_path] [model_width] [model_height] [frame_limit] [max_log_det] [device_id] [tcp|udp] [stats_interval_sec]
```

示例：

```bash
./sampleYOLOV8_npu_rtsp_process "<rtsp_url_1>,<rtsp_url_2>" ../model/yolov8s_static.om 640 640 0 3 0 tcp 5
```

640 模型示例：

```bash
./sampleYOLOV8_npu_rtsp_process "<rtsp_url_1>,<rtsp_url_2>" ../model/person_det_640_static.om 640 640 0 3 0 tcp 5
```

## RTSP Benchmark

项目提供了批量测试脚本：

```bash
cd ascend-yolov8-sample-master/src
bash bench_rtsp_models.sh
```

脚本会按模型和运行模式循环测试，并生成汇总表：

```text
src/bench_logs/summary.md
```

常用环境变量：

```bash
STREAMS="<rtsp_url_1>,<rtsp_url_2>" \
MODES="thread,process,single" \
FRAME_LIMIT=10000 \
MAX_LOG_DET=3 \
DEVICE_ID=0 \
TRANSPORT=tcp \
STATS_INTERVAL_SEC=5 \
bash bench_rtsp_models.sh
```

只跑线程和多进程：

```bash
STREAMS="<rtsp_url_1>,<rtsp_url_2>" \
MODES="thread,process" \
FRAME_LIMIT=10000 \
DEVICE_ID=0 \

```

脚本默认统计字段：

| 字段 | 说明 |
|---|---|
| `pull_decode` | 拉流和解码耗时 |
| `preprocess` | 预处理耗时 |
| `inference` | 模型推理耗时 |
| `d2h` | Device 到 Host 拷贝耗时 |
| `postprocess` | 后处理耗时 |
| `total` | 单帧总耗时 |
| `fps` | 平均帧率 |

## 日志输出

RTSP 程序会周期性打印类似信息：

```text
periodic average in last 5.00s: pull_decode=..., preprocess=..., inference=..., d2h=..., postprocess=..., total=..., fps=...
average timing(ms) for 10000 frames: pull_decode=..., preprocess=..., inference=..., d2h=..., postprocess=..., total=..., fps=...
```


