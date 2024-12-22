from __future__ import annotations
import numpy as np
from minigrad.minibuffer import MiniBuffer, UOps, dtypes, DType
from minigrad.helper import DEBUG, RETAIN_GRAD, flatten
from typing import Union, Tuple
from math import prod

class Function:
    def __init__(self, *tensors:Tensor):
        self.needs_input_grads = [t.requires_grad for t in tensors]
        self.requires_grad = any(self.needs_input_grads)
        if self.requires_grad:
            self.parents = tensors

    def forward(self):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()

    def __str__(self):
        ret = f"<{self.__class__.__name__}, requires_grad={self.requires_grad}>"
        return ret

    @classmethod
    def apply(cls, *tensors:Tensor, **kargs):
        f = cls(*tensors)
        result = Tensor(f.forward(*[t.buffer for t in tensors], **kargs), requires_grad=f.requires_grad)
        if f.requires_grad:
            result.prev_op = f
        return result

import minigrad.function as F

class Tensor:
    def __init__(self, data:Union[float, list, np.ndarray, MiniBuffer], dtype=dtypes.float32, requires_grad=False, retain_grad=False):
        if isinstance(data, MiniBuffer):
            self.buffer = data
        elif isinstance(data, np.ndarray):
            self.buffer = MiniBuffer(data)
        elif isinstance(data, float):
            self.buffer = MiniBuffer(np.array([data]))
        elif isinstance(data, list):
            self.buffer = MiniBuffer(np.array(data))
        else:
            raise NotImplementedError
        self.buffer.data = self.buffer.data.astype(dtype.np)
        self.grad = None
        self.prev_op = None
        self.requires_grad = requires_grad
        self.retain_grad = retain_grad

    @property
    def shape(self):
        return self.buffer.shape

    @staticmethod
    def zeros(shape:Union[list, tuple], dtype=dtypes.float32, **kargs):
        return Tensor(MiniBuffer.load_ops(UOps.CONST, shape, dtype, 0), **kargs)

    @staticmethod
    def ones(shape:Union[list, tuple], dtype=dtypes.float32, **kargs):
        return Tensor(MiniBuffer.load_ops(UOps.CONST, shape, dtype, 1), **kargs)

    @staticmethod
    def full(shape:Union[list, tuple], fill, dtype=dtypes.float32, **kargs):
        return Tensor(MiniBuffer.load_ops(UOps.CONST, shape, dtype, fill), **kargs)

    @staticmethod
    def rand(shape:Union[list, tuple], dtype=dtypes.float32, seed=None, **kargs):
        return Tensor(MiniBuffer.load_ops(UOps.RAND, shape, dtype, seed), **kargs)

    @staticmethod
    def randn(shape:Union[list, tuple], dtype=dtypes.float32, seed=None, **kargs):
        return Tensor(MiniBuffer.load_ops(UOps.RANDN, shape, dtype, seed), **kargs)

    @staticmethod
    def normal_xavier(shape:Union[list, tuple], dtype=dtypes.float32, seed=None, **kargs):
        assert len(shape) == 2, "Len of shape should be 2 for xavier initialization"
        buffer = MiniBuffer.load_ops(UOps.RANDN, shape, dtype, seed)
        buffer = buffer.math_ops(UOps.MUL, buffer.const(2).math_ops(UOps.DIV, buffer.const(sum(shape))).math_ops(UOps.SQRT))
        return Tensor(buffer, **kargs)

    @staticmethod
    def scaled_uniform(shape:Union[list, tuple], scale, dtype=dtypes.float32, seed=None, **kargs):
        buffer = MiniBuffer.load_ops(UOps.RAND, shape, dtype, seed)
        buffer = buffer.math_ops(UOps.SUB, buffer.const(0.5)).math_ops(UOps.MUL, buffer.const(2 * scale))
        return Tensor(buffer, **kargs)

    @staticmethod
    def get_topo(now:Tensor) -> list:
        def _get_topo(now:Tensor, nodes:list, visited:set):
            if now in visited or now.prev_op == None:
                return []
            visited.add(now)
            for son in now.prev_op.parents:
                if son not in visited:
                    _get_topo(son, nodes, visited)
            nodes.append(now)
            return nodes
        return _get_topo(now, [], set())

    def backward(self):
        assert len(self.shape) == 0 or self.shape == (1, ), "Only scalar can be backpropagated"
        nodes = self.get_topo(self)
        self.grad = Tensor(np.array([1.], dtype=np.float32))
        if DEBUG > 0:
            print("Back prop:")
        for node in reversed(nodes):
            if node.prev_op == None:
                continue
            if DEBUG > 0:
                print(f"\t{node.prev_op}\n\t\t{node.shape}")
            grads = node.prev_op.backward(node.grad.buffer)
            grads = [Tensor(g) if g is not None else None for g in grads]
            parents = node.prev_op.parents
            assert len(grads) == len(parents), "There should be as many gradients as parents"
            for g, p in zip(grads, parents):
                if DEBUG > 0:
                    print(f"\t\t{p.shape}")
                if g is None:
                    continue
                p.grad = p.grad + g if p.grad is not None else g
            if not node.retain_grad and RETAIN_GRAD == 0: # intermediate node does not require to save grad
                node.grad = None

    def item(self):
        assert len(self.shape) == 0 or self.shape == (1, ), "Only scalar can be converted to item"
        return self.buffer.data.item()

    def assign(self, x:Tensor):
        assert self.shape == x.shape
        self.buffer = x.buffer

    def detach(self):
        return Tensor(self.buffer, requires_grad=False)

    def matmul(self, w:Tensor):
        assert len(self.shape) > 0 and len(w.shape) == 2, "Wrong shape"
        assert self.shape[-1] == w.shape[0], "Dimension mismatch"
        x = self.reshape((*self.shape, 1))
        return (x * w).sum(-2)

    def reshape(self, shape:Tuple[int, ...]):
        return F.Reshape.apply(self, shape=shape)

    def permute(self, dims:Tuple[int, ...]):
        return F.Permute.apply(self, dims=dims)

    def expand(self, shape:Tuple[int, ...]):
        return F.Expand.apply(self, shape=shape)

    def squeeze(self, dims:Union[Tuple, int]):
        return F.Squeeze.apply(self, dims=dims)

    def pad(self, pad_width:Tuple[Tuple[int, int], ...]):
        return F.Pad.apply(self, pad_width=pad_width)

    def shrink(self, shrink_width:Tuple[Tuple[int, int], ...]):
        return F.Shrink.apply(self, shrink_width=shrink_width)

    def stride(self, stride:Union[Tuple[int, ...], int]):
        if isinstance(stride, int):
            stride = (stride, )
        if len(stride) != len(self.shape):
            stride = ((1, ) * (len(self.shape) - len(stride))) + stride
        output_dims = (xs // s for xs, s in zip(self.shape, stride))
        x = self.shrink([[0, xs % s] for xs, s in zip(self.shape, stride)])
        x = x.reshape(flatten(((o, s) for o, s in zip(output_dims, stride))))
        x = x.shrink(flatten(((0, 0), (0, s - 1)) for s in stride))
        x = x.squeeze(tuple(2 * i + 1 for i in range(len(stride))))
        return x
    
    def _broadcast(self, y:Tensor):
        # print("broadcast from:", self.shape, y.shape)
        x = self.reshape((*[1] * max(len(y.shape) - len(self.shape), 0), *self.shape))
        y = y.reshape((*[1] * max(len(self.shape) - len(y.shape), 0), *y.shape))
        # print("broadcast immediate:", x.shape, y.shape)
        final_shape = []
        for a, b in zip(x.shape, y.shape):
            assert not (a != b and a != 1 and b != 1), "Broacast dimension mismatch"
            final_shape.append(max(a, b))
        x = x.expand(final_shape)
        y = y.expand(final_shape)
        # print("broadcast to:", x.shape, y.shape)
        return x, y

    def sum(self, dims:Optional[Union[Tuple, int]]=None, keepdims=False):
        if dims is None:
            dims = tuple(range(len(self.shape)))
        if isinstance(dims, int):
            dims = (dims, )
        ret = F.Sum.apply(self, dims=dims)
        if not keepdims:
            ret = ret.squeeze(dims)
        return ret

    def mean(self, dims:Optional[Union[Tuple, int]]=None, keepdims=False):
        if dims is None:
            dims = tuple(range(len(self.shape)))
        if isinstance(dims, int):
            dims = (dims, )
        return self.sum(dims=dims, keepdims=keepdims) / prod([self.shape[d] for d in dims])

    def min(self, dims:Optional[Union[Tuple, int]]=None, keepdims=False):
        if dims is None:
            dims = tuple(range(len(self.shape)))
        if isinstance(dims, int):
            dims = (dims, )
        ret = F.Min.apply(self, dims=dims)
        if not keepdims:
            ret = ret.squeeze(dims)
        return ret

    def max(self, dims:Optional[Union[Tuple, int]]=None, keepdims=False):
        if dims is None:
            dims = tuple(range(len(self.shape)))
        ret = F.Max.apply(self, dims=dims)
        if not keepdims:
            ret = ret.squeeze(dims)
        return ret

    def exp(self) -> Tensor:
        return F.Exp.apply(self)

    def sqrt(self) -> Tensor:
        return F.Sqrt.apply(self)

    def log(self) -> Tensor:
        return F.Log.apply(self)

    def neg(self) -> Tensor:
        return F.Neg.apply(self)

    def relu(self) -> Tensor:
        return F.Relu.apply(self)

    def sigmoid(self) -> Tensor:
        return F.Sigmoid.apply(self)

    def softmax(self, dim:int) -> Tensor:
        return self.exp() / self.exp().sum(dim, keepdims=True)

    def sin(self) -> Tensor:
        return F.Sin.apply(self)

    def to_tensor_if_not(self, x:Union[Tensor, float]) -> Tensor:
        if isinstance(x, Tensor):
            return x
        return Tensor([x], dtype=self.buffer.dtype)

    def add(self, x:Union[Tensor, float, int]) -> Tensor:
        return F.Add.apply(*self._broadcast(self.to_tensor_if_not(x)))

    def sub(self, x:Union[Tensor, float, int]) -> Tensor:
        return F.Sub.apply(*self._broadcast(self.to_tensor_if_not(x)))

    def mul(self, x:Union[Tensor, float, int]) -> Tensor:
        return F.Mul.apply(*self._broadcast(self.to_tensor_if_not(x)))

    def div(self, x:Union[Tensor, float, int]) -> Tensor:
        return F.Div.apply(*self._broadcast(self.to_tensor_if_not(x)))

    def __str__(self):
        np_str = str(self.buffer.data).replace('\n', '\n' + ' ' * 7)
        ret = f"Tensor({np_str}, prev_op=<{self.prev_op.__class__.__name__}>, requires_grad={self.requires_grad})"
        return ret

    def __add__(self, x) -> Tensor:
        return self.add(x)

    def __mul__(self, x) -> Tensor:
        return self.mul(x)

    def __sub__(self, x) -> Tensor:
        return self.sub(x)

    def __truediv__(self, x) -> Tensor:
        return self.div(x)

    def __neg__(self) -> Tensor:
        return self.neg()

    def __matmul__(self, x) -> Tensor:
        return self.matmul(x)
