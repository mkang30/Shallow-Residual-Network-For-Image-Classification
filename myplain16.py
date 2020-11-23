import tensorflow as tf
import numpy as np
class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.batch_size = 100
        self.num_classes = 10
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # TODO: Initialize all hyperparameters
        self.epochs = 10
        self.lr = 0.001
        self.d1_size = 4*4*20
        self.d2_size = 200
        self.normal_epsilon = 0.00001

        # TODO: Initialize all trainable parameters
        self.filter1 = tf.Variable(tf.random.truncated_normal([5,5,3,16],stddev=0.1)) # strides 2,2; padding SAME -> w'=15, pooling -> w'= 8
        self.b11 = tf.Variable(tf.random.truncated_normal([16],stddev=0.1))
        self.filter2 = tf.Variable(tf.random.truncated_normal([5,5,16,20],stddev=0.1)) # strides 1,1; padding SAME -> w'=6, pooling -> w'=4
        self.b22 = tf.Variable(tf.random.truncated_normal([20],stddev=0.1))
        self.filter3 = tf.Variable(tf.random.truncated_normal([3,3,20,20],stddev=0.1))# strides 1,1; padding SAME -> w'=4
        self.b33 = tf.Variable(tf.random.truncated_normal([20],stddev=0.1))

        self.W1 = tf.Variable(tf.random.truncated_normal([4*4*20, self.d1_size], stddev=0.1))
        self.b1 = tf.Variable(tf.random.truncated_normal([self.d1_size], stddev=0.1))
        self.W2 = tf.Variable(tf.random.truncated_normal([self.d1_size, self.d2_size], stddev=0.1))
        self.b2 = tf.Variable(tf.random.truncated_normal([self.d2_size], stddev=0.1))
        self.W3 = tf.Variable(tf.random.truncated_normal([self.d2_size,self.num_classes], stddev=0.1))
        self.b3 = tf.Variable(tf.random.truncated_normal([self.num_classes], stddev=0.1))

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)
        layer1Out = tf.nn.conv2d(inputs, self.filter1, strides = [1,2,2,1],padding="SAME")
        layer1Out = tf.nn.bias_add(layer1Out,self.b11)
        mean, variance = tf.nn.moments(layer1Out,[0,1,2])
        layer1Out = tf.nn.batch_normalization(layer1Out,mean, variance,0,1,self.normal_epsilon)
        layer1Out = tf.nn.leaky_relu(layer1Out)
        layer1Out = tf.nn.max_pool(layer1Out,3,2,padding = "SAME")

        layer2Out = tf.nn.conv2d(layer1Out, self.filter2, strides = [1,1,1,1],padding="SAME")
        layer2Out = tf.nn.bias_add(layer2Out,self.b22)
        mean, variance = tf.nn.moments(layer2Out,[0,1,2])
        layer2Out = tf.nn.batch_normalization(layer2Out,mean, variance,0,1,self.normal_epsilon)
        layer2Out = tf.nn.leaky_relu(layer2Out)
        layer2Out = tf.nn.max_pool(layer2Out,2,2,padding = "SAME")

        layer3Out = conv2d(layer2Out, self.filter3, strides = [1,1,1,1],padding="SAME") if is_testing==True else tf.nn.conv2d(layer2Out, self.filter3, strides = [1,1,1,1],padding="SAME")
        layer3Out = tf.nn.bias_add(layer3Out,self.b33)
        mean, variance = tf.nn.moments(layer3Out,[0,1,2])
        layer3Out = tf.nn.batch_normalization(layer3Out,mean, variance,0,1,self.normal_epsilon)
        layer3Out = tf.nn.leaky_relu(layer3Out)
        layer3Out = tf.reshape(layer3Out,[layer3Out.shape[0],-1])

        layer4Out = tf.matmul(layer3Out,self.W1)+self.b1
        layer4Out = tf.nn.leaky_relu(layer4Out)
        if(is_testing==False):
            layer4Out = tf.nn.dropout(layer4Out, 0.3)

        layer5Out = tf.matmul(layer4Out,self.W2)+self.b2
        layer5Out = tf.nn.leaky_relu(layer5Out)
        if(is_testing==False):
            layer5Out = tf.nn.dropout(layer5Out, 0.3)

        logits = tf.matmul(layer5Out,self.W3)+self.b3

        return logits

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        Softmax is applied in this function.
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """

        losses = tf.nn.softmax_cross_entropy_with_logits(labels,logits)
        return tf.reduce_mean(losses)

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT

        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


class  Plain16(tf.keras.Model):
    def __init__(self):
        """
        This model implements a plain CNN with 16 layers
        Implementation fo the VGG-16
        """
        super(Plain16, self).__init__()

        self.batch_size = 50
        self.num_classes = 10

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

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
        self.final = tf.keras.layers.Dense(self.num_classes)


    def call(self,images):
        #images_up = self.upsample(images)
        block_1_out = self.pool(self.bn(self.block_1_2(self.bn(self.block_1_1(images)))))
        block_2_out = self.pool(self.bn(self.block_2_2(self.bn(self.block_2_1(block_1_out)))))
        block_3_out = self.pool(self.bn(self.block_3_3(self.bn(self.block_3_2(self.bn(self.block_3_1(block_2_out)))))))
        block_4_out = self.pool(self.bn(self.block_4_3(self.bn(self.block_4_2(self.bn(self.block_4_1(block_3_out)))))))
        block_5_out = self.pool(self.bn(self.block_5_3(self.bn(self.block_5_2(self.bn(self.block_5_1(block_4_out)))))))
        flat_out = self.flat(block_5_out)
        dense_1_out = self.dense_1(flat_out)
        dense_2_out = self.dense_2(dense_1_out)
        final = self.final(dense_2_out)
        return final

    def loss(self,logits, labels):
        losses = tf.nn.softmax_cross_entropy_with_logits(labels,logits)
        return tf.reduce_mean(losses)
    def accuracy(self,logits,labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def bn(self,layer_out):
        return self.batch_norm(layer_out)
    def pool(self,layer_out):
        return self.pool_layer(layer_out)
