import numpy as np
from math import floor
from model.utils import roundupdown
from model.ports import InputPort, OutputPort

class Operator:
  op_id = 0
  def __init__(self, name):
    self.name = name
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
  def __init__(self, name):
    super().__init__(name)
  
  def forward(self, input):
    self.output_port.send(input)

class Output(Operator):
  def __init__(self, name):
    super().__init__(name)

  def forward(self):
    return self.input_port[0].data

class Conv2D(Operator):
  def __init__(self, name, input_channels, num_filters, kernel_size, padding, stride, learning_rate, activation=None):
    # weight size: (F, C, K, K)
    # bias size: (F) 
    super().__init__(name)
    self.F = num_filters
    self.K = kernel_size
    self.C = input_channels
    self.activation = activation
    self.p = padding
    self.s = stride
    self.lr = learning_rate

  def forward(self):
    # input size: (C, W, H)
    # output size: (N, F ,WW, HH)
    inputs = np.int32(self.input_port[0].data)
    C = inputs.shape[0]
    W = inputs.shape[1] + 2 * self.p
    H = inputs.shape[2] + 2 * self.p

    # self.inputs = np.zeros((C, W, H), dtype=np.int32)
    # for c in range(inputs.shape[0]):
    #   self.inputs[c,:,:] = self.zero_padding(inputs[c,:,:], self.p)

    WW = floor((W - self.K)/self.s + 1)
    HH = floor((H - self.K)/self.s + 1)

    feature_maps = np.zeros((self.F, WW, HH), dtype=np.int32)
    o_multiplier = (self.i_scales*self.w_scales)/self.o_scales

    # print(self.name+":")

    for f in range(self.F):
      for w in range(WW):
        for h in range(HH):
          for c in range(C):
            for r in range(self.K):
              for s in range(self.K):
                feature_maps[f,w,h]+=(np.int32(inputs[c,w*self.s+r,h*self.s+s]) * np.int32(self.weights[f,c,r,s]))
          # print('f:{}, w:{}, h:{}'.format(f, w, h))
          # print(feature_maps[f,w,h])
          feature_maps[f,w,h] += self.bias[f]
          # print(feature_maps[f,w,h])
          feature_maps[f,w,h] -= (self.i_zero_points * np.sum(self.sums[f]))
          # print(feature_maps[f,w,h], self.i_zero_points, np.sum(self.sums[f]))
          feature_maps[f,w,h] = roundupdown(float(feature_maps[f,w,h] * o_multiplier[f]))
          # print(feature_maps[f,w,h])
          feature_maps[f,w,h] = np.int8(np.clip(feature_maps[f,w,h] + self.o_zero_points, a_min=-128, a_max=127))
    
    # for f in range(self.F):
    #   for w in range(WW):
    #     for h in range(HH):
    #       print(feature_maps[f,w,h], end= ' ')
    #   print(end='\n\n\n')

    self.output_port.send(feature_maps)

    # def backward(self, dy):
    #   C, W, H = self.inputs.shape
    #   dx = np.zeros(self.inputs.shape)
    #   dw = np.zeros(self.weights.shape)
    #   db = np.zeros(self.bias.shape)

    #   F, W, H = dy.shape
    #   for f in range(F):
    #     for w in range(0, W, self.s):
    #       for h in range(0, H, self.s):
    #         dw[f,:,:,:]+=dy[f,w,h]*self.inputs[:,w:w+self.K,h:h+self.K]
    #         dx[:,w:w+self.K,h:h+self.K]+=dy[f,w,h]*self.weights[f,:,:,:]

    #   for f in range(F):
    #     db[f] = np.sum(dy[f, :, :])

    #   self.weights -= self.lr * dw
    #   self.bias -= self.lr * db
    #   return dx

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

    for f in range(self.F):
      for c in range(self.C):
        self.sums[f,c] = np.sum(self.weights[f,c,:,:])
          
class DepthwiseConv2D(Conv2D):
  def __init__(self, name, input_channels, num_filters, kernel_size, padding, stride, learning_rate, activation=None):
    super().__init__(name, input_channels, num_filters, kernel_size, padding, stride, learning_rate, activation)
    self.sums = np.transpose(self.sums)
    self.bias = np.reshape(self.bias, newshape=(self.C, 1))

  def forward(self):
    inputs = np.int32(self.input_port[0].data)
    C = inputs.shape[0]
    W = inputs.shape[1] + 2 * self.p
    H = inputs.shape[2] + 2 * self.p

    WW = floor((W - self.K)/self.s + 1)
    HH = floor((H - self.K)/self.s + 1)

    feature_maps = np.zeros((self.C, WW, HH), dtype=np.int32)
    o_multiplier = (self.i_scales*self.w_scales)/self.o_scales

    for c in range(self.C):
      for w in range(WW):
        for h in range(HH):
          for f in range(self.F):
            for r in range(self.K):
              for s in range(self.K):
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

    for c in range(self.C):
      for f in range(self.F):
        self.sums[c,f] = np.sum(self.weights[f,c,:,:])
        
class Maxpool2D(Operator):
  def __init__(self, name, pool_size, stride):
    super().__init__(name)
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

  # def backward(self, dy):
  #   C, W, H = self.inputs.shape
  #   dx = np.zeros(self.inputs.shape)
    
  #   for c in range(C):
  #     for w in range(0, W, self.pool):
  #       for h in range(0, H, self.pool):
  #         st = np.argmax(self.inputs[c,w:w+self.pool,h:h+self.pool])
  #         (idx, idy) = np.unravel_index(st, (self.pool, self.pool))
  #         dx[c, w+idx, h+idy] = dy[c, floor(w/self.pool), floor(h/self.pool)]
  #   return dx
    
class FullyConnected(Operator):
  def __init__(self, name, num_inputs, num_outputs, learning_rate, activation = None):
    super().__init__(name)
    self.F = num_outputs
    self.C = num_inputs
    self.lr = learning_rate
    self.activation = activation

  def forward(self):
    # print(self.name+":")
    inputs = np.int32(self.input_port[0].data)
    output = np.zeros((1, self.F), dtype=np.int32)
    o_multiplier = (self.i_scales*self.w_scales)/self.o_scales

    for f in range(self.F):
      output[0,f] = np.dot(np.int32(inputs), np.int32(self.weights[:,f])) + self.bias[f]
      # print(output[0,f])
      output[0,f] -= (self.i_zero_points * self.sums[f])
      # print(output[0,f], self.i_zero_points, self.sums[f])
      output[0,f] = roundupdown(float(output[0,f] * o_multiplier))
      output[0,f] = np.int8(np.clip(output[0,f] + self.o_zero_points, a_min=-128, a_max=127))
      # print(output[0,f])
    
    self.output_port.send(output)

  # def backward(self, dy):
  #   if dy.shape[0] == self.inputs.shape[0]:
  #     dy = dy.T
  #   dw = dy.dot(self.inputs)
  #   db = np.sum(dy, axis=1, keepdims=True)
  #   dx = np.dot(dy.T, self.weights.T)
  #   self.weights -= self.lr * dw.T
  #   self.bias -= self.lr * db
  #   return dx

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

    for f in range(self.F):
      self.sums[f] = np.sum(self.weights[:,f])

class ReLU(Operator):
  def __init__(self, name):
    super().__init__(name)
  
  def forward(self):
    inputs = self.input_port[0].data
    ret = inputs.copy()
    ret[ret < 0] = 0
    self.output_port.send(ret)
  
  # def backward(self, dy):
  #   dx = dy.copy()
  #   dx[self.inputs < 0] = 0
  #   return dx

class ReLU6(Operator):
  def __init__(self, name):
    super().__init__(name)

  def forward(self):
    inputs = self.input_port[0].data
    ret = inputs.copy()
    ret[ret < 0] = 0
    ret[ret > 6] = 6
    self.output_port.send(ret)
  
  # def backward(self, dy):
  #   dx = dy.copy()
  #   dx[self.inputs < 0] = 0
  #   return dx

class Add(Operator):
  def __init__(self, name):
    super().__init__(name)
    self.add_input_port(InputPort())

  def forward(self):
    # TODO: Add support for int8 quantization
    input = []
    for port in self.input_port:
      input.append(port.data)
    assert(input[0].shape == input[1].shape)
    self.output_port.send(input[0] + input[1])

  def feed(self, o_scales=None, o_zero_points=None, i_scalesA=None, i_zero_pointsA=None,
            w_scales=None, w_zero_points=None, i_scalesB=None, i_zero_pointsB=None):
    self.o_scales = o_scales
    self.o_zero_points = o_zero_points
    self.W_scales = w_scales
    self.w_zero_points = w_zero_points
    self.i_scalesA = i_scalesA
    self.i_zero_pointsA = i_zero_pointsA
    self.i_scalesB = i_scalesB
    self.i_zero_pointsB = i_zero_pointsB
    
class Softmax(Operator):
  def __init__(self, name=None):
    super().__init__(name)
  
  def forward(self):
    # print("softmax:")
    inputs = self.input_port[0].data
    out = np.zeros((1, inputs.shape[1]), dtype=np.int32)

    exp_table = np.exp(inputs)
    sum_exp = np.sum(exp_table)
    # print("sum_exp={}".format(sum_exp))
    inv_sum_exp = 1/(sum_exp * self.o_scales)
    # print("inv_sum_exp={}".format(inv_sum_exp))

    for i in range(inputs.shape[1]):
      prob_scaled = float(exp_table[0,i] * inv_sum_exp)
      # print("prob_scaled={}".format(prob_scaled))
      prob_quantized = roundupdown(prob_scaled) + self.i_zero_points
      # print("prob_quantized={}".format(prob_quantized))
      out[0,i] = np.clip(prob_quantized, a_min=-128, a_max=127)
      # print("output={}".format(self.out[0, i]))
    self.output_port.send(out)
  
  # def backward(self, dy):
  #   return self.out.T - dy.reshape(dy.shape[0],1)
  
  def feed(self, o_scales=None, o_zero_points=None, i_scales=None, i_zero_points=None):
    self.o_scales = o_scales
    self.o_zero_points = o_zero_points
    self.i_scales = i_scales
    self.i_zero_points = -i_zero_points

class Reshape(Operator):
  def __init__(self, name):
    super().__init__(name)

  def forward(self):
    inputs = self.input_port[0].data
    inshape_product = 1
    outshape_product = 1

    id = np.where(self.newshape == -1)
    # There must be exactly one dimension that can be inferred
    assert(len(id) == 1)

    if len(id) == 1:
      for dim in inputs.shape:
        inshape_product *= dim
      
      self.newshape[id[0]] = 1
      for dim in self.newshape:
        outshape_product *= dim

      # The quotient must be zero so that output dimensions are an exact reshape of the input dimensions
      assert(inshape_product%outshape_product == 0)
      self.newshape[id[0]] = inshape_product/outshape_product
    self.output_port.send(np.reshape(inputs, tuple(self.newshape)))

  def feed(self, newshape):
    self.newshape = newshape

class Pack(Operator):
  def __init__(self, name, axis, count):
    super().__init__(name)
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
  def __init__(self, name, axis):
    super().__init__(name)
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
  def __init__(self, name):
    super().__init__(name)
  
  def forward(self):
    self.output_port.send(1 / (1 + np.exp(-self.input_port[0].data)))

class Quantize(Operator):
  def __init__(self, name):
    super().__init__(name)

  def forward(self):
    self.output_port.send(np.clip(roundupdown(self.input_port[0].data / self.scale + self.zero_point), a_min=-128, a_max=127).astype(np.int8))

  def feed(self, scale, zero_point):
    self.scale = scale
    self.zero_point = zero_point

class Dequantize(Operator):
  def __init__(self, name):
    super().__init__(name)

  def forward(self):
    self.output_port.send((self.input_port[0].data - self.zero_point) * self.scale)

  def feed(self, scale, zero_point):
    self.scale = scale
    self.zero_point = zero_point