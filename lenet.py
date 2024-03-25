#!/usr/bin/env python

import numpy as np
import sys
import mnist
import pickle
from model.network import Network
from model.utils import zero_padding
from model.operators import *

def test_with_pretrained_weights(model, data, label, test_size, weights_file):
  with open(weights_file, 'rb') as handle:
    b = pickle.load(handle)
    model.layers[1].feed(b[5]['conv1.weights'], b[5]['conv1.bias'],
                        b[6]['conv1.output.scales'], b[6]['conv1.output.zero_points'],
                        b[5]['conv1.weights.scales'], b[5]['conv1.weights.zero_points'],
                        b[5]['conv1.bias.scales'], b[5]['conv1.bias.zero_points'],
                        b[0]['input.scales'], b[0]['input.zero_points'])
    model.layers[3].feed(b[4]['conv2.weights'], b[4]['conv2.bias'],
                        b[7]['conv2.output.scales'], b[7]['conv2.output.zero_points'],
                        b[4]['conv2.weights.scales'], b[4]['conv2.weights.zero_points'],
                        b[4]['conv2.bias.scales'], b[4]['conv2.bias.zero_points'],
                        b[6]['conv1.output.scales'], b[6]['conv1.output.zero_points'])
    model.layers[5].feed(b[3]['conv3.weights'], b[3]['conv3.bias'],
                        b[8]['conv3.output.scales'], b[8]['conv3.output.zero_points'],
                        b[3]['conv3.weights.scales'], b[3]['conv3.weights.zero_points'],
                        b[3]['conv3.bias.scales'], b[3]['conv3.bias.zero_points'],
                        b[7]['conv2.output.scales'], b[7]['conv2.output.zero_points'])
    model.layers[6].feed(b[9]['reshape1.newshape'])
    model.layers[7].feed(b[2]['fc1.weights'], b[2]['fc1.bias'],
                        b[10]['fc1.output.scales'], b[10]['fc1.output.zero_points'],
                        b[2]['fc1.weights.scales'], b[2]['fc1.weights.zero_points'],
                        b[2]['fc1.bias.scales'], b[2]['fc1.bias.zero_points'],
                        b[8]['conv3.output.scales'], b[8]['conv3.output.zero_points'])
    model.layers[8].feed(b[1]['fc2.weights'], b[1]['fc2.bias'],
                          b[11]['fc2.output.scales'], b[11]['fc2.output.zero_points'],
                          b[1]['fc2.weights.scales'], b[1]['fc2.weights.zero_points'],
                          b[1]['fc2.bias.scales'], b[1]['fc2.bias.zero_points'],
                          b[10]['fc1.output.scales'], b[10]['fc1.output.zero_points'])
    model.layers[9].feed(b[12]['softmax.output.scales'], b[12]['softmax.output.zero_points'],
                          b[11]['fc2.output.scales'], b[11]['fc2.output.zero_points'])
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
      if type(model.layers[l]) is Input and isinstance(model.layers[l], Operator):
        model.connect_ops(l, l+1)
        model.layers[l].forward(x)
      elif type(model.layers[l]) is Output and isinstance(model.layers[l], Operator):
        output = model.layers[l].forward()
      else:
        model.connect_ops(l, l+1)
        model.layers[l].forward()

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
  lr = 0 # learning rate
  lenet.add_op(Input(name='input'))
  lenet.add_op(Conv2D(name='conv1', input_channels=1, num_filters=6, kernel_size=5, padding=0, stride=1, learning_rate=lr))
  lenet.add_op(Maxpool2D(name='maxpool1', pool_size=2, stride=2))
  lenet.add_op(Conv2D(name='conv2', input_channels=6, num_filters=16, kernel_size=5, padding=0, stride=1, learning_rate=lr))
  lenet.add_op(Maxpool2D(name='maxpool2', pool_size=2, stride=2))
  lenet.add_op(Conv2D(name='conv3', input_channels=16, num_filters=120, kernel_size=5, padding=0, stride=1, learning_rate=lr))
  lenet.add_op(Reshape(name='reshape'))
  lenet.add_op(FullyConnected(name='fc1', num_inputs=120, num_outputs=84, learning_rate=lr))
  lenet.add_op(FullyConnected(name='fc2', num_inputs=84, num_outputs=10, learning_rate=lr))
  lenet.add_op(Softmax(name='softmax'))
  lenet.add_op(Output(name='output'))
  lenet.layer_num = len(lenet.layers)

  return lenet

if __name__=="__main__":
  np.set_printoptions(threshold=sys.maxsize)
  print('Loading data......')
  num_classes = 10
  # train_images = mnist.train_images() #[60000, 28, 28]
  # train_labels = mnist.train_labels()
  test_images = mnist.test_images()
  test_labels = mnist.test_labels()

  print('Preparing data......')
  # training_data = train_images.reshape(60000, 1, 28, 28)
  # training_labels = np.eye(num_classes)[train_labels]
  test_data = test_images.reshape(10000, 1, 28, 28)
  testing_labels = np.eye(num_classes)[test_labels]

  testing_data = np.zeros((10000, test_data.shape[1], test_data.shape[2]+4, test_data.shape[3]+4), dtype=np.int32)
  for n in range(10000):
    for c in range(testing_data.shape[1]):
      testing_data[n,c,:,:] = zero_padding(test_data[n,c,:,:], 2)

  testing_data = np.clip(np.float32(testing_data) - 128, a_max=127, a_min=-128)
  # print('Training Lenet......')
  # net.train(training_data, training_labels, 64, 10, 'weights.pkl')
  # print('Testing Lenet......')
  # net.test(testing_data, testing_labels, 100)
  model = build()
  print('Testing with pretrained weights......')
  test_with_pretrained_weights(model, testing_data, testing_labels, 10000, './params.pkl')
