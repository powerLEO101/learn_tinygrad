from minigrad.tensor import Tensor
from collections import OrderedDict
from typing import Dict, List
# https://github.com/tinygrad/tinygrad/blob/master/tinygrad/nn/state.py

def get_state_dict(obj, prefix:str='', tensor_type=Tensor) -> Dict[str, Tensor]:
  """
  Returns a state_dict of the object, with optional prefix.

  ```python exec="true" source="above" session="tensor" result="python"
  class Net:
    def __init__(self):
      self.l1 = nn.Linear(4, 5)
      self.l2 = nn.Linear(5, 6)

  net = Net()
  print(nn.state.get_state_dict(net).keys())
  ```
  """
  if isinstance(obj, tensor_type): return {prefix.strip('.'):obj}
  if hasattr(obj, '_asdict'): return get_state_dict(obj._asdict(), prefix, tensor_type)  # namedtuple
  if isinstance(obj, OrderedDict): return get_state_dict(dict(obj), prefix, tensor_type)
  if hasattr(obj, '__dict__'): return get_state_dict(obj.__dict__, prefix, tensor_type)
  state_dict = {}
  if isinstance(obj, (list, tuple)):
    for i,x in enumerate(obj): state_dict.update(get_state_dict(x, f"{prefix}{str(i)}.", tensor_type))
  elif isinstance(obj, dict):
    for k,v in obj.items(): state_dict.update(get_state_dict(v, f"{prefix}{str(k)}.", tensor_type))
  return state_dict

def get_parameters(obj) -> List[Tensor]:
  """
  ```python exec="true" source="above" session="tensor" result="python"
  class Net:
    def __init__(self):
      self.l1 = nn.Linear(4, 5)
      self.l2 = nn.Linear(5, 6)

  net = Net()
  print(len(nn.state.get_parameters(net)))
  ```
  """
  return list(get_state_dict(obj).values())
