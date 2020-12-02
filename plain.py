import tensorflow as tf
import numpy as np
from block import Block

class Plain16(tf.keras.Model):
    def __init__(self):
        """
        This model is a plain CNN with 16 layers
        The plain model follows the structure of VGG networks -
        CNNs with 3x3 by filters one after another.
        """
        super(Plain16, self).__init__()

        self.batch_size = 50
        self.num_classes = 10
        self.epochs = 20
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.block_1 = Block(64,2)
        #maxpool
        self.block_2 = Block(128,2)
        #maxpool
        self.block_3 = Block(256,4)
        #maxpool
        self.block_4 = Block(512,4)
        #maxpool
        self.block_5 = Block(512,2)
        #maxpool
        self.flat = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(512,activation="relu")
        self.dense_2 = tf.keras.layers.Dense(256,activation="relu")
        self.final = tf.keras.layers.Dense(self.num_classes, activation="softmax")


    def call(self,images,is_testing=False):
        block_1_out = tf.nn.max_pool2d(self.block_1.call(images),2,2, "VALID")
        block_2_out = tf.nn.max_pool2d(self.block_2.call(block_1_out),2,2, "VALID")
        block_3_out = tf.nn.max_pool2d(self.block_3.call(block_2_out),2,2, "VALID")
        block_4_out = tf.nn.max_pool2d(self.block_4.call(block_3_out),2,2, "VALID")
        block_5_out = tf.nn.max_pool2d(self.block_5.call(block_4_out),2,2, "VALID")

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

class Plain34(tf.keras.Model):
    def __init__(self):
        """
        This model is a plain CNN with 34 layers
        The plain model follows the structure of VGG networks -
        CNNs with 3x3 by filters one after another.
        """
        super(Plain34, self).__init__()

        self.batch_size = 50
        self.num_classes = 10
        self.epochs = 50
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.block_1 = Block(64,6)
        #avg_pool
        self.block_2 = Block(128,8)
        #avg_pool
        self.block_3 = Block(256,12)
        #avg_pool
        self.block_4 = Block(512,6)
        #avg_pool
        self.flat = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(1024,activation="relu")
        self.final = tf.keras.layers.Dense(self.num_classes, activation="softmax")


    def call(self,images,is_testing=False):
        block_1_out = tf.nn.avg_pool2d(self.block_1.call(images),2,2, "VALID")
        block_2_out = tf.nn.avg_pool2d(self.block_2.call(block_1_out),2,2, "VALID")
        block_3_out = tf.nn.avg_pool2d(self.block_3.call(block_2_out),2,2, "VALID")
        block_4_out = tf.nn.avg_pool2d(self.block_4.call(block_3_out),2,2, "VALID")
        flat_out = self.flat(block_4_out)
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
