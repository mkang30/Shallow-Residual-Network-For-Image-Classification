import tensorflow as tf
import numpy as np



class  Plain16(tf.keras.Model):
    def __init__(self):
        """
        This model implements a plain CNN with 16 layers
        Implementation fo the VGG-16
        """
        super(Plain16, self).__init__()

        self.batch_size = 100
        self.num_classes = 10

        self.optimizer = tf.keras.optimizers.Adam()

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.pool_layer = tf.keras.layers.MaxPool2D(strides=[2,2])

        self.upsample = tf.keras.layers.UpSampling2D(size=[7,7])
        self.block_1_1 = tf.keras.layers.Conv2D(64,3,padding="same",activation="relu")#224x224x64
        self.block_1_2 = tf.keras.layers.Conv2D(64,3,padding = "same", activation="relu")#224x224x64
        self.pool_1 = tf.keras.layers.MaxPool2D(strides=[2,2]) #112x112x64
        self.block_2_1 = tf.keras.layers.Conv2D(128,3,padding="same",activation="relu")#112x112x128
        self.block_2_2 = tf.keras.layers.Conv2D(128,3,padding="same",activation="relu")#112x112x128
        self.pool_2 = tf.keras.layers.MaxPool2D(strides=[2,2]) #56x56x128
        self.block_3_1 =tf.keras.layers.Conv2D(256,3,padding="same",activation="relu") #56x56x256
        self.block_3_2 =tf.keras.layers.Conv2D(256,3,padding="same",activation="relu") #56x56x256
        self.block_3_3 =tf.keras.layers.Conv2D(256,3,padding="same",activation="relu") #56x56x256
        self.pool_3 = tf.keras.layers.MaxPool2D(strides=[2,2]) #28x28x256
        self.block_4_1 = tf.keras.layers.Conv2D(512,3,padding="same",activation="relu") #28x28x512
        self.block_4_2 = tf.keras.layers.Conv2D(512,3,padding="same",activation="relu") #28x28x512
        self.block_4_3 = tf.keras.layers.Conv2D(512,3,padding="same",activation="relu") #28x28x512
        self.pool_4 = tf.keras.layers.MaxPool2D(strides=[2,2]) #14x14x512
        self.block_5_1 = tf.keras.layers.Conv2D(512,3,padding="same",activation="relu") #14x14x512
        self.block_5_2 =tf.keras.layers.Conv2D(512,3,padding="same",activation="relu") #14x14x512
        self.block_5_3 =tf.keras.layers.Conv2D(512,3,padding="same",activation="relu") #14x14x512
        self.pool_5 = tf.keras.layers.MaxPool2D(strides=[2,2]) #7x7x512
        self.flat = tf.keras.layers.Flatten();
        self.dense_1 = tf.keras.layers.Dense(2048,activation="relu")
        self.dense_2 = tf.keras.layers.Dense(2048,activation="relu")
        self.final = tf.keras.layers.Dense(self.num_classes,activation="softmax")


    def call(self,images):
        images_up = self.upsample(images)
        block_1_out = self.pool(self.bn(self.block_1_2(self.bn(self.block_1_1(images_up)))))
        block_2_out = self.pool(self.bn(self.block_2_2(self.bn(self.block_2_1(block_1_out)))))
        block_3_out = self.pool(self.bn(self.block_3_3(self.bn(self.block_3_2(self.bn(self.block_3_1(block_2_out)))))))
        block_4_out = self.pool(self.bn(self.block_4_3(self.bn(self.block_4_2(self.bn(self.block_4_1(block_3_out)))))))
        block_5_out = self.pool(self.bn(self.block_5_3(self.bn(self.block_5_2(self.bn(self.block_5_1(block_4_out)))))))
        flat_out = self.flat(block_5_out)
        dense_1_out = self.dense_1(flat_out)
        dense_2_out = self.dense_2(dense_1_out)
        final = self.final(dense_2_out)
        return final

    def loss(self,probs, labels):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels,probs))

    def accuracy(probs,labels):
        correct_predictions = tf.equal(tf.argmax(probs, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def bn(self,layer_out):
        return self.batch_norm(layer_out)
    def pool(self,layer_out):
        return self.pool_layer(layer_out)
