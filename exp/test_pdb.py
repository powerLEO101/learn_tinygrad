from tinygrad import Tensor

a = Tensor.rand(2, 2)
b = Tensor.rand(2, 2)
c = a + b

breakpoint()
print(c.numpy())
