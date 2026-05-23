atc --model=1280_s_static.onnx --framework=5 --output=1280_s_static --input_shape="images:1,3,1280,1280" --soc_version=Ascend310B1  --insert_op_conf=aipp_1280.cfg

atc --model=1280_s.onnx --framework=5 --output=1280_s --input_shape="images:1,3,1280,1280" --soc_version=Ascend310B1  --insert_op_conf=aipp_1280.cfg

atc --model=yolov8s.onnx --framework=5 --output=yolov8s --input_shape="images:1,3,640,640" --soc_version=Ascend310B1  --insert_op_conf=aipp.cfg

atc --model=yolov8s_static.onnx --framework=5 --output=yolov8s_static --input_shape="images:1,3,640,640" --soc_version=Ascend310B1  --insert_op_conf=aipp.cfg

帮我测试一下模型以下的模型
person_det.om ： 1280 m 模型
person_det_static.om ： 1280 m npu后处理 模型 

person_det_640.om ：640 m 模型
person_det_640_static.om 

1280_s.om  ： 1280 s 模型
1280_s_staic.om ： 1280 s npu后处理 模型 

yolov8s.om ：640 s 模型
yolov8s_static.om ：640 s npu后处理 模型

汇总各个过程识别时间，分别多进程 
示例命令
./sampleYOLOV8_npu_rtsp_thread "rtsp://admin:bnc123456@192.168.100.19/h264/ch1/main/av_stream,rtsp://admin:bnc123456@192.168.100.18/h264/ch1/main/av_stream" ../model/person_det.om 1280 1280 0 3 1 tcp 5

./sampleYOLOV8_npu_rtsp_process "rtsp://admin:bnc123456@192.168.100.19/h264/ch1/main/av_stream,rtsp://admin:bnc123456@192.168.100.18/h264/ch1/main/av_stream" ../model/person_det_static.om 1280 1280 0 3 1 tcp 5

以及单进程
./sampleYOLOV8_npu_rtsp "rtsp://admin:bnc123456@192.168.100.19/h264/ch1/main/av_stream" ../model/person_det_static.om 1280 1280 10000 3 1 tcp

最后形成一个表格