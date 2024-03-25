import numpy as np

def roundupdown(input):
    '''Round float input to the nearest integer value'''
    if type(input) != float:
      raise TypeError('input value must be of type float')
    else:
      if input >= 0 or input.is_integer():
        return np.round(input)
      else:
        (frac, integer) = np.modf(input)
        if abs(frac) < 0.5:
          return integer
        else:
          return integer-1
        
def zero_padding(inputs, size):
      w, h = inputs.shape[0], inputs.shape[1]
      new_w = 2 * size + w
      new_h = 2 * size + h
      out = np.zeros((new_w, new_h), dtype=np.int32)
      out[size:w+size, size:h+size] = inputs
      return out

def cross_entropy(inputs, labels):

    out_num = labels.shape[0]
    p = np.sum(labels.reshape(1,out_num)*inputs)
    loss = -np.log(p)
    return loss