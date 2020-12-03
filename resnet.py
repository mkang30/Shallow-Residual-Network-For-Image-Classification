import tensorflow as tf
import numpy as np
from ResBlock import ResBlock

class  ResNet18(tf.keras.Model):
    def __init__(self):
        """
        This model implements a plain CNN with 16 layers
        Implementation fo the VGG-16
        """
        super(ResNet18, self).__init__()

        self.batch_size = 100
        self.num_classes = 10
        self.normal_epsilon = 1e-3
        self.loss_list =[]


        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        #self.pool_layer = tf.keras.layers.MaxPool2D(strides=[2,2])
        #self.batch_norm = tf.keras.layers.BatchNormalization()

        self.block_1 = tf.keras.layers.Conv2D(16,3,strides=1,padding='same',activation="relu")#224x224x64
        self.pool_1 = tf.keras.layers.MaxPool2D(pool_size =(3,3),strides=1,padding="same") #112x112x64
        self.res_block_2_1 = ResBlock(3,16,1)
        self.res_block_2_2 = ResBlock(3,16,1)
        self.res_block_2_3 = ResBlock(3,16,1)
        self.res_block_3_1 = ResBlock(3,32,2,is_resampled=True)
        self.res_block_3_2 = ResBlock(3,32,1)
        self.res_block_3_3 = ResBlock(3,32,1)
        self.res_block_4_1 = ResBlock(3,64,2,is_resampled=True)
        self.res_block_4_2 = ResBlock(3,64,1)
        self.res_block_4_3 = ResBlock(3,64,1)
        self.res_block_5_1 = ResBlock(3,128,2,is_resampled=True)
        self.res_block_5_2 = ResBlock(3,128,1)
        self.res_block_5_3 = ResBlock(3,128,1)
        self.ave_pool_layer = tf.keras.layers.AveragePooling2D(strides=[2,2])
        self.flat = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(512,activation="relu")
        self.dense_2 = tf.keras.layers.Dense(256,activation="relu")
        self.final = tf.keras.layers.Dense(self.num_classes, activation="softmax")


    def call(self,images,is_testing=False):
        #print("images.shape",images.get_shape().as_list())
        after_conv = self.block_1(images)
        #print(" after_conv.shape",after_conv.get_shape().as_list())
        block_1_out = self.pool_1(self.bn(after_conv))
       # block_1_out = self.pool_1(self.bn(self.block_1(images)))
        #print("block_1_out.shape",block_1_out.get_shape().as_list())
        block_2_out = self.res_block_2_3(self.res_block_2_2(self.res_block_2_1(block_1_out)))
        block_3_out = self.res_block_3_3(self.res_block_3_2(self.res_block_3_1(block_2_out)))
        #block_4_out = self.res_block_4_3(self.res_block_4_2(self.res_block_4_1(block_3_out)))
        #block_5_out = self.res_block_5_3(self.res_block_5_2(self.res_block_5_1(block_4_out)))
        flat_out = self.flat(block_3_out)

        dense_1_out = self.dense_1(flat_out)
        if(is_testing==False):
            dense_1_out = tf.nn.dropout(dense_1_out, 0.3)
        dense_2_out = self.dense_2(dense_1_out)
        if(is_testing==False):
            dense_2_out = tf.nn.dropout(dense_2_out, 0.3)
        final = self.final(dense_2_out)
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

class  ResNet32(tf.keras.Model):
    def __init__(self):
        """
        This model implements a plain CNN with 16 layers
        Implementation fo the VGG-16
        """
        super(ResNet32, self).__init__()

        self.batch_size = 100
        self.num_classes = 10
        self.normal_epsilon = 1e-3
        self.loss_list =[]


        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        #self.pool_layer = tf.keras.layers.MaxPool2D(strides=[2,2])
        #self.batch_norm = tf.keras.layers.BatchNormalization()

        self.conv_0 = tf.keras.layers.Conv2D(64,3,strides=1,padding='same',activation="relu")
        self.max_pool= tf.keras.layers.MaxPool2D(pool_size =(3,3),strides=1,padding="same") 
        #input should be 32*32*16
        self.res_block_1_1 = ResBlock(3,64,1)
        self.res_block_1_2 = ResBlock(3,64,1)
        self.res_block_1_3 = ResBlock(3,64,1)
        self.res_block_1_4 = ResBlock(3,64,1)
        self.res_block_1_5 = ResBlock(3,64,1)
        #input should be 32*32*16
        self.res_block_2_1 = ResBlock(3,128,2,is_resampled=True)
        self.res_block_2_2 = ResBlock(3,128,1)
        self.res_block_2_3 = ResBlock(3,128,1)
        self.res_block_2_4 = ResBlock(3,128,1)
        self.res_block_2_5 = ResBlock(3,128,1)
        #input should be 16*16*64
        self.res_block_3_1 = ResBlock(3,256,2,is_resampled=True)
        self.res_block_3_2 = ResBlock(3,256,1)
        self.res_block_3_3 = ResBlock(3,256,1)
        self.res_block_3_4 = ResBlock(3,256,1)
        self.res_block_3_5 = ResBlock(3,256,1)
        ##input should be 8*8*64
        self.ave_pool_layer = tf.keras.layers.AveragePooling2D(strides=[2,2])
        self.flat = tf.keras.layers.Flatten()
        self.final = tf.keras.layers.Dense(self.num_classes, activation="softmax")


    def call(self,images,is_testing=False):
        #print("images.shape",images.get_shape().as_list())
        conv_0_out = self.max_pool(self.bn(self.conv_0(images)))
        #print("pool_out.shape",pool_out.get_shape().as_list())
        block_1_out = self.res_block_1_5(self.res_block_1_4(self.res_block_1_3(self.res_block_1_2(self.res_block_1_1(conv_0_out)))))
        block_2_out = self.res_block_2_5(self.res_block_2_4(self.res_block_2_3(self.res_block_2_2(self.res_block_2_1(block_1_out)))))
        block_3_out = self.res_block_3_5(self.res_block_3_4(self.res_block_3_3(self.res_block_3_2(self.res_block_3_1(block_2_out)))))
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
