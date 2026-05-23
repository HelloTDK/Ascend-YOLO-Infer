#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 YOLOv8 检测头输出 [1, 4+num_classes, num_boxes] 的 ONNX
改造成带静态 NMS 的 ONNX，输出固定为 [1, max_det, 6]：
[x1, y1, x2, y2, score, class_id]

当前脚本按 purn_1280.onnx 的形态设计：
- 输入 ONNX 输出: [1, 6, 33600] (4 + 2 classes)
- NMS 使用 class-agnostic（对 max_class_score 做 NMS）
- 输出固定 max_det 行，不足部分补零
"""

import argparse
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def make_i64(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def make_f32(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入 onnx 路径，例如 purn_1280.onnx")
    parser.add_argument("--output", required=True, help="输出 onnx 路径，例如 purn_1280_static_nms.onnx")
    parser.add_argument("--max-det", type=int, default=300, help="固定输出框数量")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="NMS score threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    args = parser.parse_args()

    model = onnx.load(args.input)
    graph = model.graph

    if len(graph.output) < 1:
        raise RuntimeError("输入 ONNX 没有输出")

    raw_out_name = graph.output[0].name
    raw_out_shape = [d.dim_value if d.dim_value != 0 else d.dim_param
                     for d in graph.output[0].type.tensor_type.shape.dim]
    # 预期 [1, C, N]
    if (len(raw_out_shape) != 3) or (raw_out_shape[0] != 1):
        raise RuntimeError(f"不支持的输出形状: {raw_out_shape}, 预期 [1, C, N]")

    c_dim = int(raw_out_shape[1]) if isinstance(raw_out_shape[1], (int, np.integer)) else None
    n_dim = int(raw_out_shape[2]) if isinstance(raw_out_shape[2], (int, np.integer)) else None
    if (c_dim is None) or (n_dim is None):
        raise RuntimeError(f"当前脚本要求输出通道和框数是静态整数，实际: {raw_out_shape}")
    if c_dim <= 4:
        raise RuntimeError(f"输出通道数异常: C={c_dim}")

    num_classes = c_dim - 4
    print(f"[INFO] raw output: {raw_out_name}, shape=[1,{c_dim},{n_dim}], num_classes={num_classes}")

    # 常量初始化器
    inits = [
        make_i64("nms_slice_boxes_starts", [0]),
        make_i64("nms_slice_boxes_ends", [4]),
        make_i64("nms_slice_cls_starts", [4]),
        make_i64("nms_slice_cls_ends", [c_dim]),
        make_i64("nms_slice_axis_c", [1]),
        make_i64("nms_slice_axis_last", [2]),
        make_i64("nms_slice_step", [1]),
        make_i64("nms_x_start", [0]),
        make_i64("nms_x_end", [1]),
        make_i64("nms_y_start", [1]),
        make_i64("nms_y_end", [2]),
        make_i64("nms_w_start", [2]),
        make_i64("nms_w_end", [3]),
        make_i64("nms_h_start", [3]),
        make_i64("nms_h_end", [4]),
        make_f32("nms_half", [0.5]),
        make_i64("nms_max_output_boxes_per_class", [args.max_det]),
        make_f32("nms_iou_threshold", [args.iou_thres]),
        make_f32("nms_score_threshold", [args.conf_thres]),
        make_i64("nms_gather_col_2", [2]),
        make_i64("nms_unsq_axis1", [1]),
        make_i64("nms_sq_axis1", [1]),
        make_i64("nms_sq_axes_batch", [0, 1]),
        make_i64("nms_gather_zero_idx", [0]),
        make_i64("nms_range_start", [0]),
        make_i64("nms_range_delta", [1]),
        make_f32("nms_zero_det_base", np.zeros((args.max_det, 6), dtype=np.float32)),
    ]
    graph.initializer.extend(inits)

    nodes = []

    # 1) 切分 boxes / class_scores
    nodes.append(helper.make_node(
        "Slice",
        [raw_out_name, "nms_slice_boxes_starts", "nms_slice_boxes_ends", "nms_slice_axis_c", "nms_slice_step"],
        ["nms_boxes_xywh_c_first"],
        name="NMS_Slice_Boxes",
    ))
    nodes.append(helper.make_node(
        "Slice",
        [raw_out_name, "nms_slice_cls_starts", "nms_slice_cls_ends", "nms_slice_axis_c", "nms_slice_step"],
        ["nms_cls_scores_c_first"],
        name="NMS_Slice_ClassScores",
    ))

    # 2) class-agnostic 分数与类别（取每个框最大类分数）
    nodes.append(helper.make_node(
        "ReduceMax",
        ["nms_cls_scores_c_first"],
        ["nms_scores_max_c_first"],
        name="NMS_ReduceMaxScores",
        axes=[1],
        keepdims=1,
    ))
    nodes.append(helper.make_node(
        "ArgMax",
        ["nms_cls_scores_c_first"],
        ["nms_cls_ids_c_first"],
        name="NMS_ArgMaxClass",
        axis=1,
        keepdims=1,
    ))

    # 3) boxes: [1,4,N] -> [1,N,4] 并从 xywh 转成 xyxy
    nodes.append(helper.make_node(
        "Transpose",
        ["nms_boxes_xywh_c_first"],
        ["nms_boxes_xywh"],
        name="NMS_Transpose_Boxes",
        perm=[0, 2, 1],
    ))

    # x,y,w,h
    nodes.append(helper.make_node(
        "Slice",
        ["nms_boxes_xywh", "nms_x_start", "nms_x_end", "nms_slice_axis_last", "nms_slice_step"],
        ["nms_x"],
        name="NMS_Slice_X",
    ))
    nodes.append(helper.make_node(
        "Slice",
        ["nms_boxes_xywh", "nms_y_start", "nms_y_end", "nms_slice_axis_last", "nms_slice_step"],
        ["nms_y"],
        name="NMS_Slice_Y",
    ))
    nodes.append(helper.make_node(
        "Slice",
        ["nms_boxes_xywh", "nms_w_start", "nms_w_end", "nms_slice_axis_last", "nms_slice_step"],
        ["nms_w"],
        name="NMS_Slice_W",
    ))
    nodes.append(helper.make_node(
        "Slice",
        ["nms_boxes_xywh", "nms_h_start", "nms_h_end", "nms_slice_axis_last", "nms_slice_step"],
        ["nms_h"],
        name="NMS_Slice_H",
    ))

    nodes.append(helper.make_node("Mul", ["nms_w", "nms_half"], ["nms_half_w"], name="NMS_HalfW"))
    nodes.append(helper.make_node("Mul", ["nms_h", "nms_half"], ["nms_half_h"], name="NMS_HalfH"))
    nodes.append(helper.make_node("Sub", ["nms_x", "nms_half_w"], ["nms_x1"], name="NMS_X1"))
    nodes.append(helper.make_node("Sub", ["nms_y", "nms_half_h"], ["nms_y1"], name="NMS_Y1"))
    nodes.append(helper.make_node("Add", ["nms_x", "nms_half_w"], ["nms_x2"], name="NMS_X2"))
    nodes.append(helper.make_node("Add", ["nms_y", "nms_half_h"], ["nms_y2"], name="NMS_Y2"))
    nodes.append(helper.make_node(
        "Concat",
        ["nms_x1", "nms_y1", "nms_x2", "nms_y2"],
        ["nms_boxes_xyxy"],
        name="NMS_Concat_XYXY",
        axis=2,
    ))

    # NMS 需要 scores: [1, num_classes_for_nms, N]，这里用 [1,1,N]（class-agnostic）
    nodes.append(helper.make_node(
        "NonMaxSuppression",
        ["nms_boxes_xyxy", "nms_scores_max_c_first",
         "nms_max_output_boxes_per_class", "nms_iou_threshold", "nms_score_threshold"],
        ["nms_selected_indices"],
        name="NMS_Core",
        center_point_box=0,
    ))

    # 4) 根据 selected_indices 收集 boxes / score / cls
    nodes.append(helper.make_node(
        "Gather",
        ["nms_selected_indices", "nms_gather_col_2"],
        ["nms_box_indices_ex"],
        name="NMS_Gather_BoxIndicesEx",
        axis=1,
    ))
    nodes.append(helper.make_node(
        "Squeeze",
        ["nms_box_indices_ex", "nms_sq_axis1"],
        ["nms_box_indices"],
        name="NMS_Squeeze_BoxIndices",
    ))

    nodes.append(helper.make_node(
        "Squeeze",
        ["nms_boxes_xyxy", "nms_gather_zero_idx"],
        ["nms_boxes_xyxy_no_batch"],
        name="NMS_Squeeze_BoxesBatch",
    ))
    nodes.append(helper.make_node(
        "Gather",
        ["nms_boxes_xyxy_no_batch", "nms_box_indices"],
        ["nms_boxes_sel"],
        name="NMS_Gather_BoxesSel",
        axis=0,
    ))

    nodes.append(helper.make_node(
        "Squeeze",
        ["nms_scores_max_c_first", "nms_sq_axes_batch"],
        ["nms_scores_vec"],
        name="NMS_Squeeze_Scores",
    ))
    nodes.append(helper.make_node(
        "Gather",
        ["nms_scores_vec", "nms_box_indices"],
        ["nms_scores_sel"],
        name="NMS_Gather_ScoresSel",
        axis=0,
    ))
    nodes.append(helper.make_node(
        "Unsqueeze",
        ["nms_scores_sel", "nms_unsq_axis1"],
        ["nms_scores_sel_col"],
        name="NMS_Unsqueeze_Scores",
    ))

    nodes.append(helper.make_node(
        "Squeeze",
        ["nms_cls_ids_c_first", "nms_sq_axes_batch"],
        ["nms_cls_ids_vec"],
        name="NMS_Squeeze_ClsIds",
    ))
    nodes.append(helper.make_node(
        "Gather",
        ["nms_cls_ids_vec", "nms_box_indices"],
        ["nms_cls_sel_i64"],
        name="NMS_Gather_ClsSel",
        axis=0,
    ))
    nodes.append(helper.make_node(
        "Cast",
        ["nms_cls_sel_i64"],
        ["nms_cls_sel_f32"],
        name="NMS_Cast_ClsToF32",
        to=TensorProto.FLOAT,
    ))
    nodes.append(helper.make_node(
        "Unsqueeze",
        ["nms_cls_sel_f32", "nms_unsq_axis1"],
        ["nms_cls_sel_col"],
        name="NMS_Unsqueeze_Cls",
    ))

    nodes.append(helper.make_node(
        "Concat",
        ["nms_boxes_sel", "nms_scores_sel_col", "nms_cls_sel_col"],
        ["nms_det_mx6"],
        name="NMS_Concat_Det",
        axis=1,
    ))

    # 5) 固定输出到 [max_det,6]：不足补零
    nodes.append(helper.make_node(
        "Shape",
        ["nms_det_mx6"],
        ["nms_det_shape"],
        name="NMS_DetShape",
    ))
    nodes.append(helper.make_node(
        "Gather",
        ["nms_det_shape", "nms_gather_zero_idx"],
        ["nms_det_rows"],
        name="NMS_DetRows",
        axis=0,
    ))
    nodes.append(helper.make_node(
        "Range",
        ["nms_range_start", "nms_det_rows", "nms_range_delta"],
        ["nms_row_indices"],
        name="NMS_RowRange",
    ))
    nodes.append(helper.make_node(
        "Unsqueeze",
        ["nms_row_indices", "nms_unsq_axis1"],
        ["nms_row_indices_2d"],
        name="NMS_RowIndices2D",
    ))
    nodes.append(helper.make_node(
        "ScatterND",
        ["nms_zero_det_base", "nms_row_indices_2d", "nms_det_mx6"],
        ["nms_det_fixed"],
        name="NMS_Scatter_FixedDet",
    ))
    nodes.append(helper.make_node(
        "Unsqueeze",
        ["nms_det_fixed", "nms_gather_zero_idx"],
        ["output0_nms"],
        name="NMS_Unsqueeze_Output",
    ))

    graph.node.extend(nodes)

    # 替换模型输出（兼容不同 protobuf/onnx 版本）
    try:
        del graph.output[:]
    except Exception:
        while len(graph.output) > 0:
            graph.output.pop()
    graph.output.append(
        helper.make_tensor_value_info("output0_nms", TensorProto.FLOAT, [1, args.max_det, 6])
    )

    # 保持 opset 13
    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print(f"[INFO] saved: {args.output}")
    print(f"[INFO] output shape: [1, {args.max_det}, 6]")


if __name__ == "__main__":
    main()
