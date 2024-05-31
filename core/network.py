from core.kernel_ops import Operator
from collections import OrderedDict

class Network:
  def __init__(self):
    self.layers = OrderedDict()
    self.layer_num = 0

  def add_op(self, op):
    self.layers[Operator.op_id] = op

  def connect_ops(self, src, dst):
    self.layers[dst].add_input_port()
    self.layers[src].output_port.connect(self.layers[dst].input_port[-1])