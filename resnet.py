from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
import math

def initialize_variable(name,shape,initializer,trainable=True):
    var = tf.get_variable(name,shape=shape,dtype=tf.float32,initializer=initializer,trainable=trainable)
    return var

class  ResNet18(tf.keras.Model):
    def __init__(self,num_classes,batch_size):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. Do not modify the constructor, as doing so 
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(ResNet34, self).__init__()

        self.batch_size = batch_size
        self.num_classes = num_classes
        # Initialize all hyperparameters
        self.learning_rate = 1e-3
        self.stride_all = 2
        # parameters for conv1
        self.conv1_num_outputs = 64
        self.conv1_filter_size = 7
        self.conv1_max_pool_size = 3
        # parameters for conv2_x
        self.conv2_filter_size =3
        self.conv2_num_outputs = 64
        self.conv2_num_layers =2
        # parameters for conv3_x
        self.conv3_filter_size =3
        self.conv3_num_outputs = 128
        self.conv3_num_layers =2 
        # parameters for conv4_x
        self.conv4_filter_size = 3
        self.conv4_num_outputs = 256
        self.conv4_num_layers = 2
        # parameters for conv5_x
        self.conv5_filter_size = 3
        self.conv5_num_outputs = 512
        self.conv5_num_layers = 2
        ## paramters for ave_pool
        self.ave_pool_size = 7
        
        # denselayer hyperparameters
        self.dense_output_size = self.num_classes 
        # TODO: Initialize all layers
        self.conv1 = tf.keras.layers.Conv2D(self.conv1_num_outputs,self.conv1_filter_size,3,padding='SAME')
        self.conv2 = block();
        self.conv3 = block;;
        self.conv4 =;
        self.conv5 = block();
        self.dense = tf.keras.layers.Dense(self.dense_output_size,activation='softmax')
        ## initialize optimizers
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
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
        
        #conv layer 1
        output_conv_1 = tf.nn.conv2d(inputs,self.filter_conv_layer1, self.stride_layer1, self.padding)
        mean_1, variance_1 = tf.nn.moments( output_conv_1,axes = [0,1,2],keep_dims=False)
        batch_norm_1 = tf.nn.batch_normalization(output_conv_1,mean_1,variance_1,offset=None, scale=None, variance_epsilon=1e-5)
        relu_1 = tf.nn.relu(batch_norm_1)
        max_pool_1 = tf.nn.max_pool(relu_1, self.conv1_max_pool_size,self.stride_all = 2,padding = 'SAME')
        # conv_2_x
        output_conv_2 = self.conv2()
        # conv_3_x
        output_conv_3 = self.conv3()
        # conv_4_x
        output_conv_4 = self.conv4()
        # conv_5_x
        output_conv_5 = self.conv5()
        # apply ave_pool
        output_after_ave_pool = tf.nn.ave_pool(output_conv_5,self.ave_pool_size = 7)
        # flatten it
        after_flatten = tf.squeeze(output_after_ave_pool,[1,2])
        ## input to dense
        probs  = self.dense(after_flatten)
        return probs
        
        

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        Softmax is applied in this function.
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        #print("logits",tf.shape(logits))
        #print("labels.shape",tf.shape(labels))
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
        mean_loss = tf.reduce_mean(losses)
        #print("loss.shape",tf.shape(loss))
        return mean_loss

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

def train(model, dir_name_train):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    dir_path_train = dir_name_train  +'/*.jpg'
    dataset = tf.data.Dataset.list_files(dir_path_train,shuffle=False)
    dataset.map(map_func=load_and_process_image,num_parallel_calls = 8)
    dataset = dataset.batch(model.batch_size,drop_remainder=True)
    dataset = dataset.prefecth(1)
    
   labels = get_labels ()
    for i, batch in enumerate(dataset):
        with tf.GradientTape() as tape:
            predictions = model.call(batch) #
            losses = model.loss(predictions,labels[i*model.batch_size,(i+1)*model.batch_size])
        gradients = tape.gradient(losses, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, dir_name_test):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    dir_path_test = dir_name_train  +'/*.jpg'
    dataset = tf.data.Dataset.list_files(dir_path_test,shuffle=False)
    dataset.map(map_func=load_and_process_image,num_parallel_calls = 8)
    dataset = dataset.batch(model.batch_size,drop_remainder=True)
    dataset = dataset.prefecth(1)
    test_labels =get_labels
    total_accuracy = tf.convert_to_tensor(0,dtype=tf.float64)
    for i, bacth in enumerate(dataset):
        logits = model.call(batch,is_testing = True)
        accuracy = model.accuracy(logits,test_labels)
        print("test accuracy",accuracy)
        total_accuracy+=accuracy 
    average_accuracy = total_accuracy/(i+1)
    return average_accuracy 


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''
    model = ResNet18()
    num_epochs =10
    train_dir =""
    test_dir =""
    for i in range(num_epochs):
        model.train(train_dir)
        accuarcy = model.test(test_dir) 
    print("accuarcy after 10 epochs:",accuaracy)   
    return


if __name__ == '__main__':
    main()
