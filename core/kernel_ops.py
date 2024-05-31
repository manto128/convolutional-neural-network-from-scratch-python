import numpy as np
from math import floor, exp
from core.utils import roundupdown
from core.ports import InputPort, OutputPort
import copy
class Operator:
  op_id = -1 # id for input operator is -1
  def __init__(self):
    self.input_port = []
    self.output_port = OutputPort()
    self.id = Operator.op_id
    Operator.op_id += 1

  def add_input_port(self):
    self.input_port.append(InputPort())

  def forward(self, *args, **kwargs):
    pass

  def backward(self, *args, **kwargs):
    pass

  def feed(self, *args, **kwargs):
    pass

class Input(Operator):
  def __init__(self):
    super().__init__()
    Operator.op_id -= 1
    self.id = Operator.op_id
  
  def forward(self, input):
    self.output_port.send(input)

class Output(Operator):
  def __init__(self):
    super().__init__()

  def forward(self):
    return self.input_port[0].data

class Conv2D(Operator):
  def __init__(self, stride=1, padding=0, activation=None):
    # weight size: (F, C, K, K)
    # bias size: (F) 
    super().__init__()
    self.p = padding
    self.s = stride
    self.activation = activation

  def forward(self):
    # input size: (C, W, H)
    # output size: (N, F ,WW, HH)
    inputs = np.int32(self.input_port[0].data)
    C = inputs.shape[0]
    W = inputs.shape[1] + 2 * self.p
    H = inputs.shape[2] + 2 * self.p
    F = self.weights.shape[0]
    K = self.weights.shape[2]

    assert(K == self.weights.shape[3])
    assert(C == self.weights.shape[1])

    WW = floor((W - K)/self.s + 1)
    HH = floor((H - K)/self.s + 1)

    feature_maps = np.zeros((F, WW, HH), dtype=np.int32)
    o_multiplier = (self.i_scales*self.w_scales)/self.o_scales

    for f in range(F):
      for w in range(WW):
        for h in range(HH):
          for c in range(C):
            for r in range(K):
              for s in range(K):
                feature_maps[f,w,h]+=(np.int32(inputs[c,w*self.s+r,h*self.s+s]) * np.int32(self.weights[f,c,r,s]))
          feature_maps[f,w,h] += self.bias[f]
          feature_maps[f,w,h] -= (self.i_zero_points * np.sum(self.sums[f]))
          feature_maps[f,w,h] = roundupdown(float(feature_maps[f,w,h] * o_multiplier[f]))
          feature_maps[f,w,h] = np.int8(np.clip(feature_maps[f,w,h] + self.o_zero_points, a_min=-128, a_max=127))

    self.output_port.send(feature_maps)

  def feed(self, weights, bias, o_scales=None, o_zero_points=None, 
            w_scales=None, w_zero_points=None, b_scales=None, b_zero_points=None,
            i_scales=None, i_zero_points=None):
    self.weights = weights
    self.bias = bias
    self.o_scales = o_scales
    self.o_zero_points = o_zero_points
    self.w_scales = w_scales
    self.w_zero_points = w_zero_points
    self.b_scales = b_scales
    self.b_zero_points = b_zero_points
    self.i_scales = i_scales
    self.i_zero_points = i_zero_points
    self.sums = np.zeros(shape=(self.weights.shape[0], self.weights.shape[1]), dtype=np.int32)

    F = self.weights.shape[0]
    C = self.weights.shape[1]
    for f in range(F):
      for c in range(C):
        self.sums[f,c] = np.sum(self.weights[f,c,:,:])
          
class DepthwiseConv2D(Conv2D):
  def __init__(self, stride=1, padding=0, activation=None):
    super().__init__(stride, padding, activation)

  def forward(self):
    inputs = np.int32(self.input_port[0].data)
    C = inputs.shape[0]
    W = inputs.shape[1] + 2 * self.p
    H = inputs.shape[2] + 2 * self.p
    F = self.weights.shape[0]
    K = self.weights.shape[2]

    assert(K == self.weights.shape[3])
    assert(C == self.weights.shape[1])

    WW = floor((W - K)/self.s + 1)
    HH = floor((H - K)/self.s + 1)

    feature_maps = np.zeros((C, WW, HH), dtype=np.int32)
    o_multiplier = (self.i_scales*self.w_scales)/self.o_scales

    for c in range(C):
      for w in range(WW):
        for h in range(HH):
          for f in range(F):
            for r in range(K):
              for s in range(K):
                feature_maps[c,w,h]+=(np.int32(inputs[c,w*self.s+r,h*self.s+s]) * np.int32(self.weights[f,c,r,s]))
      feature_maps[c,w,h] += self.bias[c]
      feature_maps[c,w,h] -= (self.i_zero_points * np.sum(self.sums[c]))
      feature_maps[c,w,h] = roundupdown(float(feature_maps[c,w,h] * o_multiplier[f]))
      feature_maps[c,w,h] = np.int8(np.clip(feature_maps[c,w,h] + self.o_zero_points, a_min=-128, a_max=127))

    self.output_port.send(feature_maps)

  def feed(self, weights, bias, o_scales=None, o_zero_points=None, 
            w_scales=None, w_zero_points=None, b_scales=None, b_zero_points=None,
            i_scales=None, i_zero_points=None):
    self.weights = weights
    self.bias = bias
    self.o_scales = o_scales
    self.o_zero_points = o_zero_points
    self.w_scales = w_scales
    self.w_zero_points = w_zero_points
    self.b_scales = b_scales
    self.b_zero_points = b_zero_points
    self.i_scales = i_scales
    self.i_zero_points = i_zero_points
    self.sums = np.zeros(shape=(self.weights.shape[1], self.weights.shape[0]), dtype=np.int32)

    F = self.weights.shape[0]
    C = self.weights.shape[1]
    for c in range(C):
      for f in range(F):
        self.sums[c,f] = np.sum(self.weights[f,c,:,:])

class FullyConnected(Operator):
  def __init__(self, activation=None):
    super().__init__()
    self.activation = activation

  def forward(self):
    inputs = np.int32(self.input_port[0].data)
    C = inputs.shape[1]
    F = self.weights.shape[1]

    assert(C == self.weights.shape[0])

    output = np.zeros((1, F), dtype=np.int32)
    o_multiplier = (self.i_scales*self.w_scales)/self.o_scales

    for f in range(F):
      output[0,f] = np.dot(np.int32(inputs), np.int32(self.weights[:,f])) + self.bias[f]
      output[0,f] -= (self.i_zero_points * self.sums[f])
      output[0,f] = roundupdown(float(output[0,f] * o_multiplier))
      output[0,f] = np.int8(np.clip(output[0,f] + self.o_zero_points, a_min=-128, a_max=127))
    
    self.output_port.send(output)

  def feed(self, weights, bias, o_scales=None, o_zero_points=None, 
            w_scales=None, w_zero_points=None, b_scales=None, b_zero_points=None,
            i_scales=None, i_zero_points=None):
    self.weights = weights
    self.bias = bias
    self.o_scales = o_scales
    self.o_zero_points = o_zero_points
    self.w_scales = w_scales
    self.w_zero_points = w_zero_points
    self.b_scales = b_scales
    self.b_zero_points = b_zero_points
    self.i_scales = i_scales
    self.i_zero_points = i_zero_points
    self.sums = np.zeros(shape=(self.weights.shape[1], 1), dtype=np.int32)

    F = self.weights.shape[1]
    for f in range(F):
      self.sums[f] = np.sum(self.weights[:,f])
        
class Maxpool2D(Operator):
  def __init__(self, pool_size, stride):
    super().__init__()
    self.pool = pool_size
    self.s = stride

  def forward(self):
    inputs = self.input_port[0].data
    C, W, H = inputs.shape
    new_width = floor((W - self.pool)/self.s + 1)
    new_height = floor((H - self.pool)/self.s + 1)
    out = np.zeros((C, new_width, new_height))
    for c in range(C):
        for w in range(new_width):
            for h in range(new_height):
                out[c, w, h] = np.max(inputs[c, w*self.s:w*self.s+self.pool, h*self.s:h*self.s+self.pool])
    
    self.output_port.send(out)

class ReLU(Operator):
  def __init__(self):
    super().__init__()
  
  def forward(self):
    inputs = self.input_port[0].data
    ret = inputs.copy()
    ret[ret < 0] = 0
    self.output_port.send(ret)

class ReLU6(Operator):
  def __init__(self):
    super().__init__()

  def forward(self):
    inputs = self.input_port[0].data
    ret = inputs.copy()
    ret[ret < 0] = 0
    ret[ret > 6] = 6
    self.output_port.send(ret)

class Add(Operator):
  def __init__(self):
    super().__init__()
    self.add_input_port(InputPort())

  def forward(self):
    input = []
    for i, port in enumerate(self.input_port):
      input.append((np.int32(port.data) + self.i_zero_points[i]) * self.i_scales[i])

    assert(input[0].shape == input[1].shape)
    self.output_port.send((input[0] + input[1])/self.o_scales - self.o_zero_points)

  def feed(self, o_scales=None, o_zero_points=None,
          i_scalesA=None, i_zero_pointsA=None,
          i_scalesB=None, i_zero_pointsB=None):
    self.o_scales = o_scales
    self.o_zero_points = o_zero_points
    self.i_scales[0] = i_scalesA
    self.i_zero_points[0] = i_zero_pointsA
    self.i_scales[1] = i_scalesB
    self.i_zero_points[1] = i_zero_pointsB
    
class Softmax(Operator):
  def __init__(self=None):
    super().__init__()
  
  def forward(self):
    inputs = self.input_port[0].data
    out = np.zeros((1, inputs.shape[1]), dtype=np.int32)

    exp_table = np.exp(inputs)
    sum_exp = np.sum(exp_table)
    inv_sum_exp = 1/(sum_exp * self.o_scales)

    for i in range(inputs.shape[1]):
      prob_scaled = float(exp_table[0,i] * inv_sum_exp)
      prob_quantized = roundupdown(prob_scaled) + self.i_zero_points
      out[0,i] = np.clip(prob_quantized, a_min=-128, a_max=127)
    self.output_port.send(out)
  
  def feed(self, o_scales=None, o_zero_points=None, i_scales=None, i_zero_points=None):
    self.o_scales = o_scales
    self.o_zero_points = o_zero_points
    self.i_scales = i_scales
    self.i_zero_points = -i_zero_points

class Reshape(Operator):
  def __init__(self):
    super().__init__()

  def forward(self):
    inputs = self.input_port[0].data
    inshape_product = 1
    outshape_product = 1

    id = np.where(self.newshape == -1)
    # There must be exactly one dimension that can be inferred
    assert(len(id) == 1)

    output_shape = copy.deepcopy(self.newshape)
    if len(id) == 1:
      for dim in inputs.shape:
        inshape_product *= dim
      
      output_shape[id[0]] = 1
      for dim in output_shape:
        outshape_product *= dim

      # The quotient must be zero so that output dimensions are an exact reshape of the input dimensions
      assert(inshape_product%outshape_product == 0)
      output_shape[id[0]] = inshape_product/outshape_product
    self.output_port.send(np.reshape(inputs, tuple(output_shape)))

  def feed(self, newshape):
    self.newshape = newshape

class Pack(Operator):
  def __init__(self, axis, count):
    super().__init__()
    self.axis = axis
    self.count = count

  def forward(self):
    inputs = self.input_port[0].data
    assert(self.axis <= len(inputs.shape))

    scaleshape_list = []
    for _ in inputs.shape:
      scaleshape_list.append(1)
    scaleshape_list.insert(self.axis, self.count)
    self.output_port.send(np.tile(inputs, tuple(scaleshape_list)))

class Concatenation(Operator):
  def __init__(self, axis):
    super().__init__()
    self.axis = axis

  def forward(self):
    assert(len(self.input_port) > 1)
    port_shape_list = [[]]
    port_shape_list[0] = list(self.input_port[0].data.shape)
    output = self.input_port[0].data
    for i in len(self.input_port):
      port_shape_list[i+1] = list(self.input_port[i+1].data.shape)
      port_shape_list.pop(self.axis)
      compare_result = [a == b for a, b in zip(port_shape_list[0], port_shape_list[i+1])]
      assert(all(compare_result) is True)
      output = np.concatenate((output, self.input_port[i+1].data), axis=self.axis)

    self.output_port.send(output)

class Logistic(Operator):
  def __init__(self):
    super().__init__()
    self.cutoff_upper = 16.619047164916992188
    self.cutoff_lower = -9
  
  def forward(self):
    input = ((self.input_port[0].data - self.i_zero_points) * self.i_scales).flatten()
    output = np.zeros_like(input)
    for i in range(input.shape[0]):
      if input[i] > self.cutoff_upper:
        output[i] = 1
      elif input[i] < self.cutoff_lower:
        output[i] = exp(input[i])
      else:
        output[i] = 1/(1 + exp(-input[i]))

    reshaped_output = output.reshape(self.input_port[0].data.shape)

    self.output_port.send(np.int8(reshaped_output/self.o_scales + self.o_zero_points))

  def feed(self, o_scales=None, o_zero_points=None,
            i_scales=None, i_zero_points=None):
    self.o_scales = o_scales
    self.o_zero_points = o_zero_points
    self.i_scales = i_scales
    self.i_zero_points = i_zero_points