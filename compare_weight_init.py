import torch
import torch.nn as nn
from minigrad.tensor import Tensor
from minigrad.nn import Linear, Conv2d

for i in range(10):
    mg_linear = Linear(999, 200)
    mg_conv2d = Conv2d(999, 64)

    t_linear = nn.Linear(999, 200)
    t_conv2d = nn.Conv2d(999, 64, 3)

    print(mg_linear.weights.buffer.data.std(), mg_linear.weights.buffer.data.mean(), mg_linear.weights.buffer.data.var())
    print(t_linear.weight.data.std().item(), t_linear.weight.data.mean().item(), t_linear.weight.data.var().item())

    print(mg_conv2d.weights.buffer.data.std(), mg_conv2d.weights.buffer.data.mean(), mg_conv2d.weights.buffer.data.var())
    print(t_conv2d.weight.data.std().item(), t_conv2d.weight.data.mean().item(), t_conv2d.weight.data.var().item())
    print("----")
