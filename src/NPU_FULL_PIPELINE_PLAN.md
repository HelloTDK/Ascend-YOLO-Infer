# Ascend 310B 全流程 NPU 化技术方案（YOLOV8）

## 1. 当前流程判断（基于现有工程）
- 前处理:
  - `ReadJpeg` 在 CPU 侧读文件。
  - `CopyImageToDevice` 把数据搬运到 DVPP 内存。
  - `JpegD` + `Resize` 在 DVPP（NPU侧媒体处理单元）执行。
- 推理:
  - `aclmdlExecute` 在 NPU 执行。
- 后处理:
  - 当前 `sampleYOLOV8.cpp` 为 CPU 侧 decode + NMS + 绘图。
- 结论:
  - 不是“纯 CPU 前处理”，而是“CPU读图 + DVPP前处理 + NPU推理 + CPU后处理”。

## 2. 目标流程（全流程尽量在 NPU 侧）
- 目标: 把 Decode/NMS 也下沉到模型图内（NPU），CPU 只负责最薄的结果消费与可视化。
- 推荐流程:
  1. CPU: 读图/调度（不可避免）
  2. DVPP: 解码 + Resize（已有）
  3. AIPP: 颜色空间转换、归一化、均值方差（模型前置）
  4. NPU 模型图: Backbone + Head + Decode + NMS
  5. CPU: 读取最终检测框并画图（不再跑 CPU NMS）

## 3. 落地路径（310B）
### 路径 A（推荐）
- 在导出/转换模型时，把后处理算子并入图中，模型直接输出 `N x 6/7/8`（例如 `x1,y1,x2,y2,score,cls`）。
- C++ 侧使用 `sampleYOLOV8_npu.cpp` 的 NPU 后处理输出分支直接解析。

### 路径 B（兼容）
- 若模型仍输出原始 `C x N`，`sampleYOLOV8_npu.cpp` 自动回退到 CPU decode + NMS（与现有行为兼容）。

### 路径 C（进一步极致）
- 使用 AscendC / 自定义算子实现 YOLO decode+nms 并接入图编译。
- 适合批量部署，但开发成本更高（算子开发、精度对齐、版本适配）。

## 4. 关键工程点
- 继续保留 DVPP 前处理，不要退回到 OpenCV CPU resize。
- 模型输出约定:
  - 全NPU后处理模型建议输出: `[1, N, 6]` 或 `[1, 6, N]`。
  - `sampleYOLOV8_npu.cpp` 已支持自动识别上述布局。
- AIPP 配置:
  - 建议与训练/导出前处理严格对齐（色彩通道、缩放、均值方差、输入布局）。

## 5. 已实现代码说明
- 新文件: `sampleYOLOV8_npu.cpp`
  - 增加“NPU后处理输出”自动识别与解析分支。
  - 若识别到输出为 `N x 6/7/8`（且候选框数在合理范围），走 NPU 后处理分支，不做 CPU NMS。
  - 否则自动回退到原始 CPU decode+NMS 分支。
- 计时:
  - 已保留并输出 `preprocess / inference / postprocess / save` 分段耗时。

## 6. 建议的迭代顺序
1. 先用 `sampleYOLOV8_npu.cpp` 验证当前模型是否已是 NPU 后处理输出。
2. 若不是，改模型导出链路，把 decode+nms 并图后再测。
3. 对齐精度（mAP）和时延（尤其是 postprocess 时间）后再切换生产。
