# yolov7-pose-e2e-trt
yolov7-pose end2end TRT实现

* 主要目的：学习onnx的操作。
* 引用2中的trt已经很好的实现了加速，但是是将decode放在了plugin中，在这个思路中我们可以学习如何使用Graph Surgeon来对onnx操作，并将plugin插入已经存在的onnx中，
    * 延申：我们可以利用onnx操作，将预处理也插入到onnx中。
* 引用2中作者说decode会产生很多op胶水节点，比较杂乱，而且trt加速时可能会出错。因此我们的目标就是简化onnx这些胶水节点。
    * 切片操作：我们替换成split
    * view中所有值添加为int
    * 常量值断开跟踪，比如grid，anchor_grid



# 引用repo
1. https://github.com/WongKinYiu/yolov7/tree/pose
2. https://github.com/nanmi/yolov7-pose