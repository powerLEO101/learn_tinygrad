from __future__ import annotations
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class DType:
  priority: int  # this determines when things get upcasted
  itemsize: int
  name: str
  np: Optional[type]  # TODO: someday this will be removed with the "remove numpy" project
  sz: int = 1
  def __repr__(self): return f"dtypes.{self.name}"

class dtypes:
    float32 = DType(0, 4, "float32", np.float32)

class UOps(Enum):
    # Unary
    NOOP = auto()
    NEG = auto()

    # Binary
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    CMPLT = auto()
    CMPBT = auto()
    CMPEQ = auto()

    # Ternary
    WHERE = auto()

    # Reduce
    MAX = auto()
    MIN = auto()
    SUM = auto()

    # Movement
    RESHAPE = auto()
    PERMUTE = auto()
    EXPAND = auto()

    # Load
    EMPTY = auto()
    CONST = auto()
    RAND = auto()

class MiniBuffer:
    def __init__(self, data:np.ndarray, dtype:DType=dtypes.float32):
        self.data = data
        self.data = self.data.astype(dtype.np)
        self.dtype = dtype

    @property
    def shape(self):
        return self.data.shape

    # @property
    # def dtype(self): # TODO a more sophisticated dtype
    #     return self.dtype

    @staticmethod
    def load_ops(op, shape, dtype:Dtype, arg=None):
        if op == UOps.EMPTY:
            ret = np.empty(shape, dtype=dtype.np)
        elif op == UOps.RAND:
            ret = np.random.default_rng(arg).random(shape, dtype=dtype.np)
        elif op == UOps.CONST:
            ret = np.full(shape, arg, dtype=dtype.np)
        else:
            raise NotImplementedError(op)
        return MiniBuffer(ret, dtype)

    def const(self, fill):
        new_buffer = MiniBuffer.load_ops(UOps.CONST, self.shape, self.dtype, fill)
        return new_buffer

    def math_ops(self, op, *sources:MiniBuffer):
        if op == UOps.NOOP:
            ret = self.data
        elif op == UOps.NEG:
            ret = -self.data
        elif op == UOps.ADD:
            ret = self.data + sources[0].data
        elif op == UOps.SUB:
            ret = self.data - sources[0].data
        elif op == UOps.MUL:
            ret = self.data * sources[0].data
        elif op == UOps.DIV:
            ret = self.data / sources[0].data
        elif op == UOps.CMPLT:
            ret = self.data < sources[0].data
        elif op == UOps.CMPBT:
            ret = self.data > sources[0].data
        elif op == Uops.CMPEQ:
            ret = self.data == sources[0].data
        elif op == UOps.WHERE:
            ret = np.where(self.data, sources[0].data, sources[1].data)
        else:
            raise NotImplementedError(op)
        return MiniBuffer(ret) # TODO explicit dtype setting

    def reduce_ops(self, op, dims):
        # assert len(self.shape) == len(new_shape), 'New shape must be of same dimension as current shape'
        # dims = tuple(i for i, (a, b) in enuemrate(zip(self.shape, new_shape)) if a != b)
        if op == UOps.MIN:
            ret = self.data.min(axis=dims, keepdims=True)
        elif op == UOps.MAX:
            ret = self.data.max(axis=dims, keepdims=True)
        elif op == UOps.SUM:
            ret = self.data.sum(axis=dims, dtype=self.data.dtype, keepdims=True)
        else:
            raise NotImplementedError(op)
        return MiniBuffer(ret)

    def movement_ops(self, op, arg):
        if op == UOps.RESHAPE:
            ret = self.data.reshape(arg)
        elif op == UOps.PERMUTE:
            ret = self.data.transpose(arg)
        elif op == UOps.EXPAND:
            ret = np.broadcast_to(self.data, arg)
        else:
            raise NotImplementedError(op)
        return MiniBuffer(ret)
