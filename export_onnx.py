#coding=utf-8
import sys
sys.path.append('./')  # to run '$ python *.py' files in subdirectories
import torch
import torch.nn as nn
import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU

# Load PyTorch model
weights = 'yolov7-w6-pose.pt'
#device = torch.device('cuda:0')
model = attempt_load(weights, map_location="cpu")  # load FP32 model

# Update model
for k, m in model.named_modules():
    m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    if isinstance(m, models.common.Conv):  # assign export-friendly activations
        if isinstance(m.act, nn.Hardswish):
            m.act = Hardswish()
        elif isinstance(m.act, nn.SiLU):
            m.act = SiLU()
model.model[-1].export = False # 为fasle的话，就将输出的三个xcat。否则不会将输出cat
model.eval()

# Input
img = torch.randn(1, 3, 960, 960)  # image size(1,3,320,192) iDetection
torch.onnx.export(model, img, 'yolov7-w6-pose.onnx', verbose=False, opset_version=12, input_names=['images'], output_names=["ouputs"])