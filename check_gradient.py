from minigrad.tensor import Tensor

def approx_grad(x, fun, d=0.00001):
    assert len(x.shape) == 0 or x.shape == (1,), "Can only check vector gradient"
    y0 = fun(x - d)
    y1 = fun(x + d)
    return (y1 - y0) / (d * 2)

def fun(x):
    # return x * 10
    return ((x * 10 + 3).sqrt().log() / 2).exp()

a = Tensor.rand((1, ), requires_grad=True)
y = fun(a)
y.backward()

print(f"x: {a.item()}\ny: {y.item()}\nx_grad: {a.grad.item()}\nx_approx_grad:{approx_grad(a, fun).item()}")
