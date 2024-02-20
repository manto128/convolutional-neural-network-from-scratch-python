import numpy as np
import sys
import mnist
from model.network import Net

np.set_printoptions(threshold=sys.maxsize)

print('Loading data......')
num_classes = 10
# train_images = mnist.train_images() #[60000, 28, 28]
# train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

print('Preparing data......')
# train_images -= int(np.mean(train_images))
# train_images = train_images / int(np.std(train_images))
# test_images -= int(np.mean(test_images))
max = np.float32(np.max(test_images))
min = np.float32(np.min(test_images))
range = np.float32(2**8-1)
test_images = np.clip(np.round(np.float32(test_images) / ((max - min) / range)) - 128, a_max=127, a_min=-127)
# training_data = train_images.reshape(60000, 1, 28, 28)
# training_labels = np.eye(num_classes)[train_labels]
testing_data = test_images.reshape(10000, 1, 28, 28)
testing_labels = np.eye(num_classes)[test_labels]
# testing_labels = test_labels.reshape(1, 10000)

net = Net()
# print('Training Lenet......')
# net.train(training_data, training_labels, 64, 10, 'weights.pkl')
# print('Testing Lenet......')
# net.test(testing_data, testing_labels, 100)
print('Testing with pretrained weights......')
net.test_with_pretrained_weights(testing_data, testing_labels, 100, './params.pkl')