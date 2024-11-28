from minigrad.tensor import Function
from minigrad.minibuffer import MiniBuffer
from minigrad.minibuffer import UOps
from typing import Tuple

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
        return (self.y.math_ops(UOps.MUL, grad) if self.needs_input_grads[0] else None,\
                self.x.math_ops(UOps.MUL, grad.math_ops(UnaryOps.NEG)).math_ops(UOps.DIV, self.y.math_ops(UOps.MUL, self.y)) if self.needs_input_grads[1] else None)

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

class Sum(Function):
    def forward(self, x:MiniBuffer, dims:Tuple[int, ...]):
        self.x, self.dims = x, dims
        ret = self.x.reduce_ops(UOps.SUM, dims)
        return ret

    def backward(self, grad:MiniBuffer):
        return grad.movement_ops(UOps.EXPAND, self.x.shape)

class Min(Function):
    def forward(self, x:MiniBuffer, dims:Tuple[int, ...]):
        self.x, self.ret, self.dims = x, x.reduce_ops(UOps.MIN, dims), dims
        return self.ret
    
    def backward(self, grad:MiniBuffer):
        mask = self.x.const(1).math_ops(UOps.SUB, self.x.math_ops(UOps.CMPBT, self.ret))
        mask = mask.math_ops(UOps.DIV, mask.reduce_ops(UOps.SUM, dims))
        return grad.movement_ops(UOps.EXPAND, self.x.shape).math_ops(UOps.MUL, mask)

class Max(Function):
    def forward(self, x:MiniBuffer, dims:Tuple[int, ...]):
        self.x, self.ret, self.dims = x, x.reduce_ops(UOps.MAX, dims), dims
        return self.ret
    
    def backward(self, grad:MiniBuffer):
        mask = self.x.const(1).math_ops(UOps.SUB, self.x.math_ops(UOps.CMPLT, self.ret))
        mask = mask.math_ops(UOps.DIV, mask.reduce_ops(UOps.SUM, dims))
        return grad.movement_ops(UOps.EXPAND, self.x.shape).math_ops(UOps.MUL, mask)
