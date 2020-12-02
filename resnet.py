import tensorflow as tf
import numpy as np

from block import ResBlock

class ResNet16(tf.keras.Model):
    def __init__(self):
        """
        This model is a Residual Neural Network with 16 layers.
        The structure is the same as the Plain16 with the only
        difference of skip connections wih the identity function.

        """
        super(ResNet16, self).__init__()

        self.batch_size = 50
        self.num_classes = 10
        self.normal_epsilon = 1e-3
        self.epochs = 20
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.res_block_1 = ResBlock(64,64,is_resampled=True)
        #maxpool
        self.res_block_2 = ResBlock(128,128,is_resampled=True)
        #maxpool
        self.res_block_3_1 = ResBlock(256,256,is_resampled=True)
        self.res_block_3_2 = ResBlock(256,256)
        #maxpool
        self.res_block_4_1 = ResBlock(512,512,is_resampled=True)
        self.res_block_4_2 = ResBlock(512,512)
        #maxpool
        self.res_block_5 = ResBlock(512,512,is_resampled=True)
        self.flat = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(512,activation="relu")
        self.dense_2 = tf.keras.layers.Dense(256,activation="relu")
        self.final = tf.keras.layers.Dense(self.num_classes, activation="softmax")

    def call(self,images,is_testing=False):

        block_1_out = tf.nn.max_pool2d(self.res_block_1(images),2,2, "VALID")
        block_2_out = tf.nn.max_pool2d(self.res_block_2(block_1_out),2,2, "VALID")
        block_3_out = tf.nn.max_pool2d(self.res_block_3_2(self.res_block_3_1(block_2_out)),2,2, "VALID")
        block_4_out = tf.nn.max_pool2d(self.res_block_4_2(self.res_block_4_1(block_3_out)),2,2, "VALID")
        block_5_out = tf.nn.max_pool2d(self.res_block_5(block_4_out),2,2, "VALID")

        flat_out = self.flat(block_5_out)
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

class ResNet34(tf.keras.Model):
    def __init__(self):
        """
        This model is a Residual Neural Network with  layers.
        The structure is the same as the Plain17 with the only
        difference of skip connections wih the identity function.
        """
        super(ResNet34, self).__init__()

        self.batch_size = 50
        self.num_classes = 10
        self.normal_epsilon = 1e-3
        self.epochs = 50
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.res_block_1_1 = ResBlock(64,64,is_resampled=True)
        self.res_block_1_2 = ResBlock(64,64)
        self.res_block_1_3 = ResBlock(64,64)
        #avg_pool
        self.res_block_2_1 = ResBlock(128,128,is_resampled=True)
        self.res_block_2_2 = ResBlock(128,128)
        self.res_block_2_3 = ResBlock(128,128)
        self.res_block_2_4 = ResBlock(128,128)
        #avg_pool
        self.res_block_3_1 = ResBlock(256,256,is_resampled=True)
        self.res_block_3_2 = ResBlock(256,256)
        self.res_block_3_3 = ResBlock(256,256)
        self.res_block_3_4 = ResBlock(256,256)
        self.res_block_3_5 = ResBlock(256,256)
        self.res_block_3_6 = ResBlock(256,256)
        #avg_pool
        self.res_block_4_1 = ResBlock(512,512,is_resampled=True)
        self.res_block_4_2 = ResBlock(512,512)
        self.res_block_4_3 = ResBlock(512,512)
        #avg_pool

        self.flat = tf.keras.layers.Flatten();
        self.dense_1 = tf.keras.layers.Dense(1024,activation="relu")
        self.final = tf.keras.layers.Dense(self.num_classes, activation="softmax")

    def call(self,images,is_testing=False):
        block_1_out =self.res_block_1_3(self.res_block_1_2(self.res_block_1_1(images)))
        block_1_out_pool = tf.nn.avg_pool2d(block_1_out,2,2, "VALID")
        block_2_out = self.res_block_2_4(self.res_block_2_3(self.res_block_2_2(self.res_block_2_1(block_1_out_pool))))
        block_2_out_pool = tf.nn.avg_pool2d(block_2_out,2,2, "VALID")
        block_3_out = self.res_block_3_6(self.res_block_3_5(self.res_block_3_4(self.res_block_3_3(self.res_block_3_2(self.res_block_3_1(block_2_out_pool))))))
        block_3_out_pool = tf.nn.avg_pool2d(block_3_out,2,2, "VALID")
        block_4_out = self.res_block_4_3(self.res_block_4_2(self.res_block_4_1(block_3_out_pool)))
        block_4_out_pool = tf.nn.avg_pool2d(block_4_out,2,2, "VALID")

        flat_out = self.flat(block_4_out_pool)
        dense_1_out = self.dense_1(flat_out)
        if(is_testing==False):
            dense_1_out = tf.nn.dropout(dense_1_out, 0.3)
        final = self.final(dense_1_out)
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
