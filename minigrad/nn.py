from minigrad.tensor import Tensor
from minigrad.helper import flatten
from typing import List, Union, Tuple, Iterator, Optional
from math import prod


# Optimizers
class Optimizer:
    def __init__(self, params:List[Tensor], lr:float=0.001):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None
            p.prev_op = None

class SGD(Optimizer):
    def step(self):
        for p in self.params:
            p.assign(p - p.grad * self.lr)

# Functionals

def linear(x:Tensor, w:Tensor, b:Tensor) -> Tensor:
    return x @ w + b

def _pool(x:Tensor, kernel:Tuple, stride:Union[Tuple, int]=1, pad:Union[Tuple, int]=1, dilation:Union[Tuple, int]=1) -> Tensor:
    if isinstance(stride, int):
        stride = tuple(stride for _ in range(len(kernel)))
    if isinstance(dilation, int):
        dilation = tuple(dilation for _ in range(len(kernel)))
    if isinstance(pad, int):
        pad = [[0, 0]] * (len(x.shape) - len(kernel)) + [[pad, pad] for _ in range(len(kernel))]
    x = x.pad(pad)
    original_x_shape, prefix_shape = x.shape, list(x.shape[:-len(kernel)])
    x = x.reshape(prefix_shape + flatten([(1, s) for s in x.shape[-len(kernel):]])) # create a new empty dim before each dim we want to pool
    x = x.expand(prefix_shape + flatten([(k, s) for k, s in zip(kernel, original_x_shape[-len(kernel):])])) # expand the new dim to the size of kernel
    x = x.reshape(prefix_shape + [k * s for k, s in zip(kernel, original_x_shape[-len(kernel):])]) # flatten
    x = x.pad([[0, 0]] * len(prefix_shape) + [[0, k * d] for k, d in zip(kernel, dilation)])
    x = x.reshape(prefix_shape + flatten([(k, s + d) for k, d, s in zip(kernel, dilation, original_x_shape[-len(kernel):])])) # each line is off by dilation, so we can directly access the next element
    x = x.shrink([[0, 0]] * len(prefix_shape) + flatten([([0, 0], [0, k * d]) for k, d in zip(kernel, dilation)]))
    x = x.stride([1] * len(prefix_shape) + flatten([[1, s] for s in stride]))
    x = x.permute(dims=list(range(len(prefix_shape))) + [len(prefix_shape) + 2 * i + 1 for i in range(len(kernel))] + [len(prefix_shape) + 2 * i for i in range(len(kernel))])
    return x

def max_pool2d(x:Tensor, kernel:Union[int, Tuple], stride=None, pad=1):
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if stride is None:
        stride = kernel
    x = _pool(x, kernel, stride=stride, pad=pad)
    x = x.max(tuple(range(-len(kernel), 0)))
    return x

def avg_pool2d(x:Tensor, kernel:Union[int, Tuple], stride=None, pad=1):
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if stride is None:
        stride = kernel
    x = _pool(x, kernel, stride=stride, pad=pad)
    x = x.mean(tuple(range(-len(kernel), 0)))
    return x

def conv2d(x:Tensor, w:Tensor, b:Optional[Tensor]=None, stride=1, pad=1, dilation=1):
    assert b is None or len(b.shape) == 1, "Bias has to be one dimensional"
    assert len(w.shape) == 4, "Weights have to be four dimensional, (in, out, H, W)"
    x = _pool(x, w.shape[2:], stride=stride, pad=pad, dilation=dilation) # (B, C, H, W, K0, K1)
    x = x.reshape(x.shape[:-2] + (1,) + x.shape[-2:]) # (B, C, H0, W0, 1, K0, K1)
    w = w.reshape((w.shape[0], 1, 1, *w.shape[1:])) # (in, 1, 1, out, K0, K1)
    x = (x * w).sum((-1, -2, -6)) # (B, H0, W0, C0)
    x = x.permute(dims=tuple(range(len(x.shape) - 3)) + (-1, -3, -2)) # (..., C0, H0, W0)
    if b is not None:
        x = x + b.reshape((b.shape[0], 1, 1))
    return x

def mse_loss(y_pred:Tensor, y_true:Tensor, reduce='mean') -> Tensor:
    ret = (y_pred - y_true)
    ret = (ret * ret)
    if reduce == 'mean':
        return ret.mean()
    elif reduce == 'sum':
        return ret.sum()
    else:
        raise NotImplementedError(reduce)

def cross_entropy_loss(y_pred:Tensor, y_true:Tensor, reduce='mean') -> Tensor:
    y_pred = y_pred.softmax(-1)
    ret = y_pred.log() * y_true
    if reduce == 'mean':
        return -ret.mean()
    elif reduce == 'sum':
        return -ret.sum()
    else:
        raise NotImplementedError(reduce)


# Modules

class Linear:
    def __init__(self, input_dim:int, output_dim:int):
        # self.weights = Tensor.normal_xavier((input_dim, output_dim), requires_grad=True)
        self.weights = Tensor.scaled_uniform((input_dim, output_dim), (1 / input_dim) ** 0.5, requires_grad=True)
        self.bias = Tensor.scaled_uniform((output_dim, ), (1 / input_dim) ** 0.5, requires_grad=True)

    def __call__(self, x:Tensor) -> Tensor:
        return linear(x, self.weights, self.bias)

class Conv2d:
    def __init__(self, input_dim:int, output_dim:int, bias=True, kernel_size:Union[Tuple[int, ...], int]=3, pad:Union[Tuple, int]=1, stride:Union[Tuple, int]=1, dilation:Union[Tuple, int]=1):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.weights = Tensor.scaled_uniform((input_dim, output_dim, *kernel_size), (1 / (input_dim * prod(kernel_size))) ** 0.5, requires_grad=True)
        self.bias = Tensor.scaled_uniform((output_dim, ), (1 / (input_dim * prod(kernel_size))) ** 0.5, requires_grad=True) if bias else None
        self.pad = pad
        self.stride = stride
        self.dilation = dilation

    def __call__(self, x:Tensor) -> Tensor:
        return conv2d(x, self.weights, self.bias, self.stride, self.pad, self.dilation)
