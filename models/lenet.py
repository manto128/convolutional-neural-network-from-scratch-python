#!/usr/bin/env python
import os
import sys
import numpy as np
import pickle
from core.network import Network
from core.utils import zero_padding
from core.kernel_ops import *
from core.postprocess_ops import *
from mnist import *

def test_with_pretrained_weights(model, data, label, test_size, weights_file):
  with open(weights_file, 'rb') as handle:
    compute_graph = pickle.load(handle)

  for op_idx, op in enumerate(compute_graph):
    if op['op_code'] == 'CONV_2D' \
      or op['op_code'] == 'FULLY_CONNECTED':
      model.layers[op_idx].feed(op['w_data'],
                                op['b_data'],
                                op['o_scale'],
                                op['o_zero_point'],
                                op['w_scale'],
                                op['w_zero_point'],
                                op['b_scale'],
                                op['b_zero_point'],
                                op['i_scale'],
                                op['i_zero_point'])
    elif op['op_code'] == 'RESHAPE':
      model.layers[op_idx].feed(op['new_shape'])
    elif op['op_code'] == 'SOFTMAX':
      model.layers[op_idx].feed(op['o_scale'],
                                op['o_zero_point'],
                                op['i_scale'],
                                op['i_zero_point'])
    # model.layers[0].feed(b[5]['conv1.weights'], b[5]['conv1.bias'],
    #                     b[6]['conv1.output.scales'], b[6]['conv1.output.zero_points'],
    #                     b[5]['conv1.weights.scales'], b[5]['conv1.weights.zero_points'],
    #                     b[5]['conv1.bias.scales'], b[5]['conv1.bias.zero_points'],
    #                     b[0]['input.scales'], b[0]['input.zero_points'])
    # model.layers[2].feed(b[4]['conv2.weights'], b[4]['conv2.bias'],
    #                     b[7]['conv2.output.scales'], b[7]['conv2.output.zero_points'],
    #                     b[4]['conv2.weights.scales'], b[4]['conv2.weights.zero_points'],
    #                     b[4]['conv2.bias.scales'], b[4]['conv2.bias.zero_points'],
    #                     b[6]['conv1.output.scales'], b[6]['conv1.output.zero_points'])
    # model.layers[4].feed(b[3]['conv3.weights'], b[3]['conv3.bias'],
    #                     b[8]['conv3.output.scales'], b[8]['conv3.output.zero_points'],
    #                     b[3]['conv3.weights.scales'], b[3]['conv3.weights.zero_points'],
    #                     b[3]['conv3.bias.scales'], b[3]['conv3.bias.zero_points'],
    #                     b[7]['conv2.output.scales'], b[7]['conv2.output.zero_points'])
    # model.layers[5].feed(b[9]['reshape1.newshape'])
    # model.layers[6].feed(b[2]['fc1.weights'], b[2]['fc1.bias'],
    #                     b[10]['fc1.output.scales'], b[10]['fc1.output.zero_points'],
    #                     b[2]['fc1.weights.scales'], b[2]['fc1.weights.zero_points'],
    #                     b[2]['fc1.bias.scales'], b[2]['fc1.bias.zero_points'],
    #                     b[8]['conv3.output.scales'], b[8]['conv3.output.zero_points'])
    # model.layers[7].feed(b[1]['fc2.weights'], b[1]['fc2.bias'],
    #                       b[11]['fc2.output.scales'], b[11]['fc2.output.zero_points'],
    #                       b[1]['fc2.weights.scales'], b[1]['fc2.weights.zero_points'],
    #                       b[1]['fc2.bias.scales'], b[1]['fc2.bias.zero_points'],
    #                       b[10]['fc1.output.scales'], b[10]['fc1.output.zero_points'])
    # model.layers[8].feed(b[12]['softmax.output.scales'], b[12]['softmax.output.zero_points'],
    #                       b[11]['fc2.output.scales'], b[11]['fc2.output.zero_points'])
  toolbar_width = 100
  sys.stdout.write("[%s]" % (" " * (toolbar_width-1)))
  sys.stdout.flush()
  sys.stdout.write("\b" * (toolbar_width))
  step = float(test_size)/float(toolbar_width)
  st = 1
  total_acc = 0
  for i in range(test_size):
    if i == round(step):
      step += float(test_size)/float(toolbar_width)
      st += 1
      sys.stdout.write(".")
      #sys.stdout.write("%s]a"%(" "*(toolbar_width-st)))
      #sys.stdout.write("\b" * (toolbar_width-st+2))
      sys.stdout.flush()
    x = data[i]
    y = label[i]

    for l in range(model.layer_num):
      if type(model.layers[l-1]) is Input and isinstance(model.layers[l-1], Operator):
        model.connect_ops(l-1, l)
        model.layers[l-1].forward(x)
      elif type(model.layers[l-1]) is Output and isinstance(model.layers[l-1], Operator):
        output = model.layers[l-1].forward()
      else:
        model.connect_ops(l-1, l)
        model.layers[l-1].forward()

    if np.argmax(output) == np.argmax(y):
      total_acc += 1
  sys.stdout.write("\n")
  print('=== Test Size:{0:d} === Test Acc:{1:.4f}% ==='.format(test_size, float(total_acc)/float(test_size)))

# ------------------------------------------------
  # Lenet Architecture
  # ------------------------------------------------
  # input: 28x28
  # conv1: (5x5x6)@s1p2 -> 28x28x6 {(28-5+2x2)/1+1}
  # maxpool2: (2x2)@s2 -> 14x14x6 {(28-2)/2+1}
  # conv3: (5x5x16)@s1p0 -> 10x10x16 {(14-5)/1+1}
  # maxpool4: (2x2)@s2 -> 5x5x16 {(10-2)/2+1}
  # conv5: (5x5x120)@s1p0 -> 1x1x120 {(5-5)/1+1}
  # fc6: 120 -> 84
  # fc7: 84 -> 10
  # softmax: 10 -> 10
  # ------------------------------------------------

def build():
  lenet = Network()
  lenet.add_op(Input())
  lenet.add_op(Conv2D())
  lenet.add_op(Maxpool2D(pool_size=2, stride=2))
  lenet.add_op(Conv2D())
  lenet.add_op(Maxpool2D(pool_size=2, stride=2))
  lenet.add_op(Conv2D())
  lenet.add_op(Reshape())
  lenet.add_op(FullyConnected())
  lenet.add_op(FullyConnected())
  lenet.add_op(Softmax())
  lenet.add_op(Output())
  lenet.layer_num = len(lenet.layers)

  return lenet

if __name__=="__main__":
  np.set_printoptions(threshold=sys.maxsize)
  dataset = "mnist"
  dataset_dir = os.path.join(os.path.join(os.getcwd(), "../../datasets"), dataset)
  print('Loading data......')
  num_classes = 10
  test_images = test_images(dataset_dir)
  test_labels = test_labels(dataset_dir)

  print('Preparing data......')
  test_data = test_images.reshape(10000, 1, 28, 28)
  testing_labels = np.eye(num_classes)[test_labels]

  testing_data = np.zeros((10000, test_data.shape[1], test_data.shape[2]+4, test_data.shape[3]+4), dtype=np.int32)
  for n in range(10000):
    for c in range(testing_data.shape[1]):
      testing_data[n,c,:,:] = zero_padding(test_data[n,c,:,:], 2)

  testing_data = np.clip(np.float32(testing_data) - 128, a_max=127, a_min=-128)
  model = build()
  print('Testing with pretrained weights......')
  test_with_pretrained_weights(model, testing_data, testing_labels, 100, './lenet5.pkl')
