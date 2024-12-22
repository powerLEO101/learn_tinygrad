from minigrad.tensor import Function
from minigrad.minibuffer import MiniBuffer
from minigrad.minibuffer import UOps
from minigrad.helper import argsort
from typing import Tuple, Union
import math

class Add(Function):
    def forward(self, x:MiniBuffer, y:MiniBuffer):
        self.x, self.y = x, y
        return x.math_ops(UOps.ADD, y)

    def backward(self, grad:MiniBuffer):
        return (grad if self.needs_input_grads[0] else None,\
                grad if self.needs_input_grads[1] else None)

class Sub(Function):
    def forward(self, x:MiniBuffer, y:MiniBuffer):
        self.x, self.y = x, y
        return x.math_ops(UOps.SUB, y)

    def backward(self, grad:MiniBuffer):
        return (grad if self.needs_input_grads[0] else None,\
                grad.math_ops(UOps.NEG) if self.needs_input_grads[1] else None)

class Mul(Function):
    def forward(self, x:MiniBuffer, y:MiniBuffer):
        self.x, self.y = x, y
        return x.math_ops(UOps.MUL, y)

    def backward(self, grad:MiniBuffer):
        return (self.y.math_ops(UOps.MUL, grad) if self.needs_input_grads[0] else None,\
                self.x.math_ops(UOps.MUL, grad) if self.needs_input_grads[1] else None)

class Div(Function):
    def forward(self, x:MiniBuffer, y:MiniBuffer):
        self.x, self.y = x, y
        return x.math_ops(UOps.DIV, y)

    def backward(self, grad:MiniBuffer):
        return (grad.math_ops(UOps.DIV, self.y) if self.needs_input_grads[0] else None,\
                self.x.math_ops(UOps.MUL, grad.math_ops(UOps.NEG)).math_ops(UOps.DIV, self.y.math_ops(UOps.MUL, self.y)) if self.needs_input_grads[1] else None)

class Neg(Function):
    def forward(self, x:MiniBuffer):
        self.x = x
        return x.math_ops(UOps.NEG, x)

    def backward(self, grad:MiniBuffer):
        return (self.x.math_ops(UOps.NEG, grad) if self.needs_input_grads[0] else None,)

class Noop(Function):
    def forward(self, x:MiniBuffer):
        self.x = x
        return x

    def backward(self, grad:MiniBuffer):
        return (grad, )

class Sqrt(Function):
    def forward(self, x:MiniBuffer):
        self.x, self.ret = x, x.math_ops(UOps.SQRT, x)
        return self.ret

    def backward(self, grad:MiniBuffer):
        return (grad.math_ops(UOps.DIV, self.x.const(2).math_ops(UOps.MUL, self.ret)), )

class Exp(Function):
    def forward(self, x:MiniBuffer):
        self.ret = x.math_ops(UOps.EXP, x)
        return self.ret

    def backward(self, grad:MiniBuffer):
        return (grad.math_ops(UOps.MUL, self.ret), )

class Log(Function):
    def forward(self, x:MiniBuffer):
        self.x = x
        return x.math_ops(UOps.LOG, x)

    def backward(self, grad:MiniBuffer):
        return (grad.math_ops(UOps.DIV, self.x), )

class Sin(Function):
    def forward(self, x:MiniBuffer):
        self.x = x
        return x.math_ops(UOps.SIN, x)

    def backward(self, grad:MiniBuffer):
        return (self.x.const(math.pi / 2).math_ops(UOps.SUB, self.x).math_ops(UOps.SIN).math_ops(UOps.MUL, grad), )

class Relu(Function):
  def forward(self, x:MiniBuffer) -> MiniBuffer:
      self.x = x;
      return x.math_ops(UOps.CMPBT, x.const(0)).math_ops(UOps.MUL, x)

  def backward(self, grad:MiniBuffer):
      return (self.x.math_ops(UOps.CMPBT, self.x.const(0)).math_ops(UOps.MUL, grad), )

class Sigmoid(Function):
    def forward(self, x:MiniBuffer):
        self.ret = x.const(1).math_ops(UOps.DIV, x.math_ops(UOps.NEG).math_ops(UOps.EXP).math_ops(UOps.ADD, x.const(1)))
        return self.ret

    def backward(self, grad:MiniBuffer):
        return (grad.math_ops(UOps.MUL, self.ret.math_ops(UOps.MUL, self.ret.const(1).math_ops(UOps.SUB, self.ret))), )

class Sum(Function):
    def forward(self, x:MiniBuffer, dims:Union[int, Tuple[int, ...]]):
        self.x, self.dims = x, dims
        ret = self.x.reduce_ops(UOps.SUM, dims)
        return ret

    def backward(self, grad:MiniBuffer):
        return (grad.movement_ops(UOps.EXPAND, self.x.shape), )

class Min(Function):
    def forward(self, x:MiniBuffer, dims:Union[int, Tuple[int, ...]]):
        self.x, self.ret, self.dims = x, x.reduce_ops(UOps.MIN, dims), dims
        return self.ret
    
    def backward(self, grad:MiniBuffer):
        mask = self.x.const(1).math_ops(UOps.SUB, self.x.math_ops(UOps.CMPBT, self.ret))
        mask = mask.math_ops(UOps.DIV, mask.reduce_ops(UOps.SUM, self.dims))
        return (grad.movement_ops(UOps.EXPAND, self.x.shape).math_ops(UOps.MUL, mask), )

class Max(Function):
    def forward(self, x:MiniBuffer, dims:Union[int, Tuple[int, ...]]):
        self.x, self.ret, self.dims = x, x.reduce_ops(UOps.MAX, dims), dims
        return self.ret
    
    def backward(self, grad:MiniBuffer):
        mask = self.x.const(1).math_ops(UOps.SUB, self.x.math_ops(UOps.CMPLT, self.ret))
        mask = mask.math_ops(UOps.DIV, mask.reduce_ops(UOps.SUM, self.dims))
        return (grad.movement_ops(UOps.EXPAND, self.x.shape).math_ops(UOps.MUL, mask), )

class Reshape(Function):
    def forward(self, x:MiniBuffer, shape:Tuple[int, ...]):
        self.x, self.shape = x, shape
        return x.movement_ops(UOps.RESHAPE, shape)

    def backward(self, grad:MiniBuffer):
        return (grad.movement_ops(UOps.RESHAPE, self.x.shape), )

class Permute(Function):
    def forward(self, x:MiniBuffer, dims:Tuple[int, ...]):
        assert len(dims) == len(x.shape), "Permute dims must have same length as input shape"
        dims = list(i if i >= 0 else i + len(x.shape) for i in dims)
        self.x, self.dims = x, dims
        return x.movement_ops(UOps.PERMUTE, dims)

    def backward(self, grad:MiniBuffer):
        return (grad.movement_ops(UOps.PERMUTE, argsort(self.dims)), )

class Squeeze(Function):
    def forward(self, x:MiniBuffer, dims:Union[int, Tuple[int, ...]]):
        self.x = x
        tmp = x.movement_ops(UOps.SQUEEZE, dims)
        return tmp

    def backward(self, grad:MiniBuffer):
        return (grad.movement_ops(UOps.RESHAPE, self.x.shape), )

class Expand(Function):
    def forward(self, x:MiniBuffer, shape:Tuple[int, ...]):
        self.x, self.shape = x, shape
        return x.movement_ops(UOps.EXPAND, shape)

    def backward(self, grad:MiniBuffer):
        dims = tuple(i for i, (a, b) in enumerate(zip(self.x.shape, grad.shape)) if a != b)
        return (grad.reduce_ops(UOps.SUM, dims), )

class Shrink(Function):
    def forward(self, x:MiniBuffer, shrink_width:Tuple[Tuple[int, int], ...]):
        assert len(x.shape) == len(shrink_width)
        self.x, self.pad_width = x, shrink_width
        shrink_width = tuple((a, s - b) for (a, b), s in zip(shrink_width, x.shape))
        return x.movement_ops(UOps.SLICE, shrink_width)

    def backward(self, grad:MiniBuffer):
        return (grad.movement_ops(UOps.PAD, self.pad_width), )

class Pad(Function):
    def forward(self, x:MiniBuffer, pad_width:Tuple[Tuple[int, int], ...]):
        assert len(x.shape) == len(pad_width), "Pad width must have same length as input shape"
        self.x, self.pad_width = x, pad_width
        return x.movement_ops(UOps.PAD, pad_width)
    
    def backward(self, grad:MiniBuffer):
        shrink_width = tuple((a, s - b) for (a, b), s in zip(self.pad_width, grad.shape))
        return (grad.movement_ops(UOps.SLICE, shrink_width), )
