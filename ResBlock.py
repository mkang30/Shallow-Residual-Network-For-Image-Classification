import tensorflow as tf
import numpy as np

"""
This class represents a Residual block that uses
skip connections through 2 CNNs.
"""
class ResBlock(tf.keras.layers.Layer):

    def __init__(self, filter_size_1, filter_size_2, is_resampled = False):
        """
        params:
        filter sizes are needed to constrcut CNNs
        is_pool flag accounts for the situation when pooling layer is between two CNNs
        is_resampled flag accounts for situation when filter map channels are not equal for x and f(x)
        """
        super(ResBlock,self).__init__()
        self.is_resampled = is_resampled
        self.conv_1 = tf.keras.layers.Conv2D(filter_size_1,3,padding="same",activation="relu")#224x224x64
        self.conv_2 = tf.keras.layers.Conv2D(filter_size_2,3,padding="same",activation="relu")
        self.conv_res = None
        if(is_resampled):
            self. conv_res = tf.keras.layers.Conv2D(filter_size_2,1,strides=[1,1],padding="valid")

    def call(self, inputs):
        conv_1_out = self.bn(self.conv_1(inputs))
        conv_2_out = self.bn(self.conv_2(conv_1_out))
        inputs_x = inputs
        if(self.is_resampled):
            inputs_x = self.bn(self.conv_res(inputs))
        return tf.nn.relu(inputs_x+conv_2_out)


    def bn(self, inp):
        """
        This function performs batch normalization
        """
        mean, variance = tf.nn.moments(inp,[0,1,2])
        output = tf.nn.batch_normalization(inp,mean, variance,0,1,1e-3)
        return output
