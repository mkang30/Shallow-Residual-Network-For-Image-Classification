import tensorflow as tf
import numpy as np

def bn(inp):
    """
    This function performs batch normalization
    """
    mean, variance = tf.nn.moments(inp,[0,1,2])
    output = tf.nn.batch_normalization(inp,mean, variance,0,1,1e-3)
    return output
"""
This class represents a Residual block that uses
skip connections through 2 CNNs.
"""
class ResBlock2(tf.keras.layers.Layer):

    def __init__(self, filter_size_1, filter_size_2, strides=(1,1), is_resampled = False):
        """
        This class is a residual block
        params:
        filter sizes are needed to constrcut CNNs
        is_pool flag accounts for the situation when pooling layer is between two CNNs
        is_resampled flag accounts for situation when filter map channels are not equal for x and f(x)
        """
        super(ResBlock2,self).__init__()
        self.filter_size_2 = filter_size_2
        self.is_resampled = is_resampled
        self.conv_1 = tf.keras.layers.Conv2D(filter_size_1,3,padding="same",activation="relu")#224x224x64
        self.conv_2 = tf.keras.layers.Conv2D(filter_size_2,3,padding="same")
        self.conv_res = None
        if(is_resampled):
            #1x1 convolution acts as an identity upsampling function
            self.conv_res = tf.keras.layers.Conv2D(filter_size_2,1,strides=[1,1],padding="valid")


    def call(self, inputs):
        input_channels = inputs.get_shape()[-1]
        conv_1_out = bn(self.conv_1(inputs))
        conv_2_out = bn(self.conv_2(conv_1_out))
        inputs_x = inputs
        if(self.is_resampled):
            inputs_x = bn(self.conv_res(inputs))
        return tf.nn.relu(bn(inputs_x+conv_2_out))

class ResBlock(tf.keras.layers.Layer):

    def __init__(self, filter_size, out_channel, stride=1, is_resampled = False):
        """
        params:
        filter sizes are needed to constrcut CNNs
        is_pool flag accounts for the situation when pooling layer is between two CNNs
        is_resampled flag accounts for situation when filter map channels are not equal for x and f(x)
        """
        super(ResBlock,self).__init__()
        self.is_resampled = is_resampled
        if stride == 2:
            self.conv_1 = tf.keras.layers.Conv2D(out_channel,filter_size, strides=stride,padding="same",activation="relu")
        else:
            self.conv_1 = tf.keras.layers.Conv2D(out_channel,filter_size, strides=1,padding="same",activation="relu")
        self.conv_2 = tf.keras.layers.Conv2D(out_channel,filter_size,strides=1, padding="same",activation="relu")

    def call(self, inputs):
        #print("inputs.shape",inputs.get_shape().as_list())
        input_channels = inputs.get_shape()[-1]
        conv_1_out = self.bn(self.conv_1(inputs))
        conv_2_out = self.bn(self.conv_2(conv_1_out))
        #print("conv_2_out.shape",conv_2_out.get_shape().as_list())
        if(self.is_resampled):
            pool_input = tf.nn.avg_pool(inputs,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
            #print("pool_input.shape",pool_input.get_shape().as_list())
            padded_input = tf.pad(pool_input,[[0,0],[0,0],[0,0],[input_channels//2, input_channels//2]])
            #print("padded_input.shape",padded_input.get_shape().as_list())
        else:
            padded_input = inputs
            #print("padded_input.shape",padded_input.get_shape().as_list())
        return tf.nn.relu(conv_2_out+padded_input)


    def bn(self, inp):
        """
        This function performs batch normalization
        """
        mean, variance = tf.nn.moments(inp,[0,1,2])
        output = tf.nn.batch_normalization(inp,mean, variance,0,1,1e-3)
        return output

class Block(tf.keras.layers.Layer):
    def __init__(self, filter_size, num_layers):
        """
        This class is a plain block that initializes and forward passes
        through 3x3, stride = (1,1), activation = relu, padding = same
        convolutoinal layers that don't change dimensionality
        """
        super(Block,self).__init__()
        self.layers = []
        for i in range(num_layers):
            self.layers.append(tf.keras.layers.Conv2D(filter_size,3,padding="same",activation="relu"))

    def call(self,inputs):
        output = inputs
        for i in range(len(self.layers)):
            output = bn(self.layers[i](output))
        return output
