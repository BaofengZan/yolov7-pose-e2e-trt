#coding=utf-8

"""
如果我们想将预处理加到onnx头部
1. 我们首先要导出预处理的onnx
2. 然后合并两个onnx

注意此时的输入就变成了 [n h w c]
"""

import torch
import onnx
import onnx.helper as helper


"""
yolov7-pose预处理只有 BGR->RGB   HWC->CHW  /255   

"""
class Preprocess(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 输入的为uint8的NHWC形式
    def forward(self, x):
        # x原本是uint8的先转为float
        x = x[...,[2, 1, 0]]  # 产生了Gather节点。  BGR->RGB
        x = x / 255.0
        x = x.permute(0, 3, 1, 2) # 
        return x

pre = Preprocess()
# 这里输入名字，尽量自定义，后面转trt可控
torch.onnx.export(pre, (torch.zeros((1, 960, 960, 3), dtype=torch.float),), "pre.onnx", input_names=["images"])




# 
pre = onnx.load("./pre.onnx")
model = onnx.load("../yolov7-w6-pose.onnx")

# 先把pre模型名字加上前缀
# for n in pre.graph.node:
#     print(n.name) # 节点名
#     for i in range(len(n.input)): # 一个节点可能有多个输入
#         print(n.input[i]) # 打印输入名字
#     for i in range(len(n.output)): 
#         print(n.output[i]) # 打印输出名字  

for n in pre.graph.node:
    n.name = f"pre/{n.name}"
    for i in range(len(n.input)): # 一个节点可能有多个输入
        n.input[i]= f"pre/{n.input[i]}"
    for i in range(len(n.output)): 
        n.output[i]= f"pre/{n.output[i]}"

# 2 修改另一个模型的信息
# 查看大模型的第一层名字，这里是  
# Slice_4  Slice_14  Slice_24  Slice_34
for n in model.graph.node:
    if n.name == "Slice_4":
        # 将conv_0的输入由原本的image 变为 pre的输出  pre/8
        n.input[0] = "pre/" + pre.graph.output[0].name
    if n.name == "Slice_14":
        # 将conv_0的输入由原本的image 变为 pre的输出  pre/8
        n.input[0] = "pre/" + pre.graph.output[0].name
    if n.name == "Slice_24":
        # 将conv_0的输入由原本的image 变为 pre的输出  pre/8
        n.input[0] = "pre/" + pre.graph.output[0].name
    if n.name == "Slice_34":
        # 将conv_0的输入由原本的image 变为 pre的输出  pre/8
        n.input[0] = "pre/" + pre.graph.output[0].name

for n in pre.graph.node:
    model.graph.node.append(n)


#还要将pre的输入信息 NHWC等拷贝到输入
model.graph.input[0].CopyFrom(pre.graph.input[0])
# 此时model的输入需要变为 pre的输入 pre/0
model.graph.input[0].name = "pre/" + pre.graph.input[0].name

onnx.save(model, "../merge.onnx")
