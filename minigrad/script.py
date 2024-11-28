from minigrad.tensor import Tensor
import numpy as np

a = Tensor.rand((1))
b = Tensor.rand((1))
# c = a * a
c = (-a) * a + b * a + b
print(a)
print(b)
print(c)
c.backward()
print(c.grad)
print(a.grad)
print(b.grad)
