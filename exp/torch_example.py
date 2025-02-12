import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List
import matplotlib.pyplot as plt

def target_function(x:Tensor) -> Tensor:
    return (x * x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(20, 200)
        self.l2 = nn.Linear(200, 100)
        self.l3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.l1(x).relu()
        # x = x * x
        x = self.l2(x).relu()
        # x = x * x
        x = self.l3(x)
        return x
        
model = Model()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

print(model.l1.weight.data.std(), model.l1.weight.data.mean(), model.l1.weight.data.var())

import sys
sys.exit(0)

EPOCH = 2000
for i in range(EPOCH):
    input1 = torch.rand((40, 1))
    target = target_function(input1)
    output = model(input1)
    loss_value = loss_fn(output, target)
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    print(loss_value.item())
    if i % 200 == 0:
        plt.scatter(input1.detach().numpy(), target.detach().numpy())
        plt.scatter(input1.detach().numpy(), output.detach().numpy())
        plt.show()
