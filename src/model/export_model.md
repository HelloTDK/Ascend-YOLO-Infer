atc --model=1280_s_static.onnx --framework=5 --output=1280_s_static --input_shape="images:1,3,1280,1280" --soc_version=Ascend310B1  --insert_op_conf=aipp_1280.cfg

atc --model=1280_s.onnx --framework=5 --output=1280_s --input_shape="images:1,3,1280,1280" --soc_version=Ascend310B1  --insert_op_conf=aipp_1280.cfg

atc --model=yolov8s.onnx --framework=5 --output=yolov8s --input_shape="images:1,3,640,640" --soc_version=Ascend310B1  --insert_op_conf=aipp.cfg

atc --model=yolov8s_static.onnx --framework=5 --output=yolov8s_static --input_shape="images:1,3,640,640" --soc_version=Ascend310B1  --insert_op_conf=aipp.cfg

