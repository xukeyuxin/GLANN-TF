import tensorflow as tf
import numpy as np
import os

class VGG19(object):
    def __init__(self):
        self.para = np.load(os.path.join('data','vgg19.npy'), encoding="latin1",allow_pickle=True).item()

    def __call__(self,inputs,conv3 = False, conv4 = False, conv5 = False,include_all = False):
        def relu(input, alpha=0.2):
            return tf.maximum(input, alpha * input)
        def conv(input, weight, bias):
            conv2d = tf.nn.conv2d(input, filter=weight, strides=[1, 1, 1, 1], padding='SAME')
            conv2d = tf.nn.bias_add(conv2d, bias)
            return conv2d

        def max_pooling(input):
            return tf.nn.max_pool(input, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        inputs = tf.reverse(inputs, [-1]) - np.array([103.939, 116.779, 123.68])
        ### (256,256,3)
        inputs = relu(conv(inputs, self.para["conv1_1"][0], self.para["conv1_1"][1]))
        inputs = relu(conv(inputs, self.para["conv1_2"][0], self.para["conv1_2"][1]))
        inputs = max_pooling(inputs) ## (128,128,3)
        inputs = relu(conv(inputs, self.para["conv2_1"][0], self.para["conv2_1"][1]))
        inputs = relu(conv(inputs, self.para["conv2_2"][0], self.para["conv2_2"][1]))
        inputs = max_pooling(inputs) ## (64,64,3)
        inputs = relu(conv(inputs, self.para["conv3_1"][0], self.para["conv3_1"][1]))
        inputs = relu(conv(inputs, self.para["conv3_2"][0], self.para["conv3_2"][1]))
        inputs = relu(conv(inputs, self.para["conv3_3"][0], self.para["conv3_3"][1]))
        inputs = relu(conv(inputs, self.para["conv3_4"][0], self.para["conv3_4"][1]))
        _conv3 = inputs
        if(conv3):
            return inputs
        inputs = max_pooling(inputs)  ## (32,32,3)
        inputs = relu(conv(inputs, self.para["conv4_1"][0], self.para["conv4_1"][1]))
        inputs = relu(conv(inputs, self.para["conv4_2"][0], self.para["conv4_2"][1]))
        inputs = relu(conv(inputs, self.para["conv4_3"][0], self.para["conv4_3"][1]))
        inputs = relu(conv(inputs, self.para["conv4_4"][0], self.para["conv4_4"][1]))
        _conv4 = inputs
        if(conv4):
            return inputs
        inputs = max_pooling(inputs)  ## (16,16,3)
        inputs = relu(conv(inputs, self.para["conv5_1"][0], self.para["conv5_1"][1]))
        inputs = relu(conv(inputs, self.para["conv5_2"][0], self.para["conv5_2"][1]))
        inputs = relu(conv(inputs, self.para["conv5_3"][0], self.para["conv5_3"][1]))
        inputs = relu(conv(inputs, self.para["conv5_4"][0], self.para["conv5_4"][1]))
        _conv5 = inputs
        if(conv5):
            return inputs
        if(include_all):
            return [_conv3, _conv4, _conv5]
        inputs = max_pooling(inputs)  ## (8,8,3)
        return inputs