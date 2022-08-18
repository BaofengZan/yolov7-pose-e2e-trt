# yolov7-pose-e2e-trt
yolov7-pose end2end TRT实现

* 主要目的：学习onnx的操作。

* 引用2中的trt已经很好的实现了加速，但是是将decode放在了plugin中，在这个思路中我们可以学习如何使用Graph Surgeon来对onnx操作，并将plugin插入已经存在的onnx中，
    * 延申：我们可以利用onnx操作，将预处理也插入到onnx中。
        * 我们单独增加文件实现，方便和之前的文件对比

* 引用2中作者说decode会产生很多op胶水节点，比较杂乱，而且trt加速时可能会出错。因此我们的目标就是简化onnx这些胶水节点。 具体改动请看`model/yolo.py`中IKeypoint类的forward函数。
    * 切片操作：我们替换成split
    * view中所有值添加为int
    * 常量值断开跟踪，比如grid，anchor_grid
    * 不要使用inplace操作


# 使用流程
1. 运行export_onnx.py导出onnx

    1.1 如果需要将预处理也嵌入到onnx中，运行 `onnx_add_preprocess\export_preprocess.py`脚本，最后会合并出merge.onnx
2.  生成engine
```python
trtexec --onnx=./yolov7-pose.onnx --saveEngine=./yolov7-pose_fp16.engine --fp16 --workspace=1000
```

3. 运行trt_py.py进行推理。
    
    3.1 如果预处理在onnx中，运行 trt_py_onnxpre.py。

# TODO
1. nms插入到onnx中。
    * 因为trt官方提供的nms插件，只支持box，因此若想将nms插入到onnx，只能使用插件了。
    * 我们这里使用[Tensorrt-pro](https://github.com/shouxieai/tensorRT_Pro/blob/2c5db6a987be9ab92abdc78db3ce4305053364f0/src/application/app_yolo/yolo_decode.cu)的实现


# 引用repo
1. https://github.com/WongKinYiu/yolov7/tree/pose
2. https://github.com/nanmi/yolov7-pose