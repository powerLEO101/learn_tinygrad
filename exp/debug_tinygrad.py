from tinygrad import Tensor

breakpoint()
a = Tensor.rand(10, 10)
b = Tensor.rand(10, 10)

c = a + b

print(c.numpy())
