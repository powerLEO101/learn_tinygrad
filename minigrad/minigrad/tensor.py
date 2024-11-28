from __future__ import annotations
import numpy as np
from minigrad.minibuffer import MiniBuffer, UOps, dtypes, DType
from minigrad.helper import DEBUG
from typing import Union, Tuple

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
        ret = f"Function {self.__class__.__name__}, requires_grad={self.requires_grad}"
        return ret

    @classmethod
    def apply(cls, *tensors:Tensor):
        f = cls(*tensors)
        result = Tensor(f.forward(*[t.buffer for t in tensors]))
        if f.requires_grad:
            result.prev_op = f
        return result

import minigrad.function as F

class Tensor:
    def __init__(self, data:Union[float, list, np.ndarray, MiniBuffer], dtype=dtypes.float32, requires_grad=True):
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

    @staticmethod
    def zeros(shape:Union[list, tuple], dtype=dtypes.float32, **kargs):
        return Tensor(MiniBuffer.load_ops(UOps.CONST, shape, dtype, 0), **kargs)

    @staticmethod
    def ones(shape:Union[list, tuple], dtype=dtypes.float32, **kargs):
        return Tensor(MiniBuffer.load_ops(UOps.CONST, shape, dtype, 1, **kargs))

    @staticmethod
    def full(shape:Union[list, tuple], fill, dtype=dtypes.float32, **kargs):
        return Tensor(MiniBuffer.load_ops(UOps.CONST, shape, dtype, fill), **kargs)

    @staticmethod
    def rand(shape:Union[list, tuple], dtype=dtypes.float32, **kargs):
        return Tensor(MiniBuffer.load_ops(UOps.RAND, shape, dtype), **kargs)

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
        nodes = self.get_topo(self)
        self.grad = Tensor(np.array([1.], dtype=np.float32))
        if DEBUG > 0:
            print("Back prop:")
        for node in reversed(nodes):
            if node.prev_op == None:
                continue
            if DEBUG > 0:
                print(f"\t{node.prev_op}\n{node}")
            grads = node.prev_op.backward(node.grad.buffer)
            grads = [Tensor(g) if g is not None else None for g in grads]
            parents = node.prev_op.parents
            assert len(grads) == len(parents), "There should be any many gradients as parents"
            for g, p in zip(grads, parents):
                if DEBUG > 0:
                    print(f"{p}")
                if g is None:
                    continue
                p.grad = p.grad + g if p.grad is not None else g

    def sum(self, dims:Tuple[int, ...]):
        return F.Sum.apply(self, dims)

    def min(self, dims:Tuple[int, ...]):
        return F.Min.apply(self, dims)

    def max(self, dims:Tuple[int, ...]):
        return F.Max.apply(self, dims)

    def add(self, x:Union[Tensor, float]) -> Tensor:
        return F.Add.apply(self, x)

    def neg(self):
        return F.Neg.apply(self)

    def sub(self, x:Union[Tensor, float]) -> Tensor:
        return F.Sub.apply(self, x)

    def mul(self, x:Union[Tensor, float]) -> Tensor:
        return F.Mul.apply(self, x)

    def div(self, x:Union[Tensor, float]) -> Tensor:
        return F.Div.apply(self, x)

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

    def __div__(self, x) -> Tensor:
        return self.div(x)

    def __neg__(self) -> Tensor:
        return self.neg()
