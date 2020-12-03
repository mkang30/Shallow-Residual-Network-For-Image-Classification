import tensorflow as tf
import numpy as np
from block import BlockWrapper

class  Plain20(tf.keras.Model):
    def __init__(self):
        """
        This model implements a plain CNN with 16 layers
        Implementation fo the VGG-16
        """
        super(Plain20, self).__init__()

        self.batch_size = 100
        self.num_classes = 10
        self.normal_epsilon = 1e-3
        self.loss_list =[]


        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.general_pool_layer = tf.keras.layers.MaxPool2D(strides=[2,2])
        self.conv_0 = tf.keras.layers.Conv2D(16,3,strides=1,padding='same',activation="relu")
        self.max_pool= tf.keras.layers.MaxPool2D(pool_size =(3,3),strides=1,padding="same") 
        #input should be 32*32*16
        self.block_1 = BlockWrapper(16,3,6,1)
        #input should be 32*32*16
        self.block_2 = BlockWrapper(32,3,6,2)
        #input should be 16*16*64
        self.block_3 = BlockWrapper(64,3,6,2)
        ##input should be 8*8*64
        self.ave_pool_layer = tf.keras.layers.AveragePooling2D(strides=[2,2])
        self.flat = tf.keras.layers.Flatten()
        self.final = tf.keras.layers.Dense(self.num_classes, activation="softmax")


    def call(self,images,is_testing=False):
        conv_0_out = self.max_pool(self.bn(self.conv_0(images)))
        block_1_out =  self.block_1(conv_0_out)
        block_2_out = self.block_2(block_1_out)
        block_3_out = self.block_3(block_2_out)
        flat_out = self.flat(self.ave_pool_layer(block_3_out))
        final = self.final(flat_out)
        return final

    def loss(self,probs, labels):
        losses = tf.keras.losses.sparse_categorical_crossentropy(labels,probs)
        return tf.reduce_mean(losses)
    def accuracy(self,probs,labels):
        correct_predictions = tf.equal(tf.argmax(probs, 1), labels)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def bn(self, inp):
        mean, variance = tf.nn.moments(inp,[0,1,2])
        output = tf.nn.batch_normalization(inp,mean, variance,0,1,self.normal_epsilon)
        return output
        #return self.batch_norm(inp)

class  Plain32(tf.keras.Model):
    def __init__(self):
        """
        This model implements a plain CNN with 16 layers
        Implementation fo the VGG-16
        """
        super(Plain32, self).__init__()

        self.batch_size = 100
        self.num_classes = 10
        self.normal_epsilon = 1e-3
        self.loss_list =[]


        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.general_pool_layer = tf.keras.layers.MaxPool2D(strides=[2,2])
        self.conv_0 = tf.keras.layers.Conv2D(16,3,strides=1,padding='same',activation="relu")
        self.max_pool= tf.keras.layers.MaxPool2D(pool_size =(3,3),strides=1,padding="same") 
        #input should be 32*32*16
        self.block_1 = BlockWrapper(16,3,10,1)
        #input should be 32*32*16
        self.block_2 = BlockWrapper(32,3,10,2)
        #input should be 16*16*64
        self.block_3 = BlockWrapper(64,3,10,2)
        ##input should be 8*8*64
        self.ave_pool_layer = tf.keras.layers.AveragePooling2D(strides=[2,2])
        self.flat = tf.keras.layers.Flatten()
        self.final = tf.keras.layers.Dense(self.num_classes, activation="softmax")


    def call(self,images,is_testing=False):
        conv_0_out = self.max_pool(self.bn(self.conv_0(images)))
        block_1_out =  self.block_1(conv_0_out)
        block_2_out = self.block_2(block_1_out)
        block_3_out = self.block_3(block_2_out)
        flat_out = self.flat(self.ave_pool_layer(block_3_out))
        final = self.final(flat_out)
        return final

    def loss(self,probs, labels):
        losses = tf.keras.losses.sparse_categorical_crossentropy(labels,probs)
        return tf.reduce_mean(losses)
    def accuracy(self,probs,labels):
        correct_predictions = tf.equal(tf.argmax(probs, 1), labels)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def bn(self, inp):
        mean, variance = tf.nn.moments(inp,[0,1,2])
        output = tf.nn.batch_normalization(inp,mean, variance,0,1,self.normal_epsilon)
        return output
        #return self.batch_norm(inp)