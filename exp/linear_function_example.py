from minigrad.tensor import Tensor
import numpy as np
from typing import List
from minigrad.nn import SGD, Linear, mse_loss
from minigrad.state import get_state_dict, get_parameters
import matplotlib.pyplot as plt
import time

def target_function(x:Tensor) -> Tensor:
    return (x * x).exp().relu()

class Model:
    def __init__(self):
        self.l1 = Linear(1, 200)
        self.l2 = Linear(200, 100)
        self.l3 = Linear(100, 1)

    def __call__(self, x):
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        x = self.l3(x)
        return x
        

# x = Tensor.rand((100, )) - 0.5
# y = target_function(x)
#
# plt.scatter(x.buffer.data, y.buffer.data)
# plt.show()
#
# import sys
# sys.exit(0)

model = Model()
optimizer = SGD(get_parameters(model), 0.01)
loss_fn = mse_loss

print(model.l1.weights.buffer.data.std(), model.l1.weights.buffer.data.mean(), model.l1.weights.buffer.data.var())

import sys
sys.exit(0)

EPOCH = 2000
for i in range(EPOCH):
    input1 = Tensor.rand((40, 1))
    # input2 = Tensor.rand((10, 10))
    target = target_function(input1)
    output = model(input1)
    loss_value = loss_fn(output, target)
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    print(loss_value.item())
    if i % 200 == 0:
        plt.scatter(input1.buffer.data, target.buffer.data)
        plt.scatter(input1.buffer.data, output.buffer.data)
        plt.show()
    # print(loss_value.prev_op.parents[0].prev_op.parents[0])
    # print(target)
    # print(output)
    # print(model.weights.grad)
    # print("----")
