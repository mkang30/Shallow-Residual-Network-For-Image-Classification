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
    
        # TODO: Initialize all hyperparameters
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
        #print("inputs.shape",tf.shape(inputs))
        conv_1 = tf.nn.conv2d(inputs,self.filter_conv_layer1, self.stride_layer1, self.padding)
        #print("conv_1.shape",tf.shape(conv_1))
        mean_1, variance_1 = tf.nn.moments(conv_1,axes = [0,1,2])
        batch_norm_1 = tf.nn.batch_normalization(conv_1,mean_1,variance_1,offset=None, scale=None, variance_epsilon=1e-5)
        #print("batch_norm_1.shape",tf.shape(batch_norm_1))
        relu_1 = tf.nn.relu(batch_norm_1)
        max_pool_1 = tf.nn.max_pool(relu_1,self.max_pooling_ksize_layer1,self.max_pooling_strides_layer1,padding = 'SAME')
        #print("max_pool_1.shape",tf.shape(max_pool_1))
        # conv layer 2
        conv_2 = tf.nn.conv2d(max_pool_1,self.filter_conv_layer2, self.stride_layer2, self.padding)
        mean_2, variance_2 = tf.nn.moments(conv_2,axes = [0,1,2])
        batch_norm_2 = tf.nn.batch_normalization(conv_2,mean_2,variance_2,offset=None, scale=None, variance_epsilon=1e-5)
        #print("batch_norm_2.shape",tf.shape(batch_norm_2))
        relu_2 = tf.nn.relu(batch_norm_2)
        max_pool_2 = tf.nn.max_pool(relu_2,self.max_pooling_ksize_layer2,self.max_pooling_strides_layer2,padding = 'SAME')
        #print("max_pool_2.shape",tf.shape(max_pool_2))
        # conv layer 3
        if is_testing == True:
           conv_3 = conv2d(max_pool_2,self.filter_conv_layer3, self.stride_layer3, self.padding)
        else:
            conv_3 = tf.nn.conv2d(max_pool_2,self.filter_conv_layer3, self.stride_layer3, self.padding)
        mean_3, variance_3 = tf.nn.moments(conv_3,axes =[0,1,2])
        batch_norm_3 = tf.nn.batch_normalization(conv_3,mean_3,variance_3,offset=None, scale=None, variance_epsilon=1e-5)
        relu_3 = tf.nn.relu(batch_norm_3)
        #print("relu_3.shape",tf.shape(relu_3))
        logits_1 = tf.nn.relu(tf.matmul(tf.reshape(relu_3,(relu_3.numpy().shape[0],-1)),self.weights_dense_1)+self.bias_dense_1)
        logits_1 = tf.nn.dropout(logits_1 ,0.3)
        logits_2 = tf.nn.relu(tf.matmul(logits_1,self.weights_dense_2)+self.bias_dense_2)
        logits_2 = tf.nn.dropout(logits_2,0.3)
        logits_3 = tf.matmul(logits_2,self.weights_dense_3)+self.bias_dense_3
        return logits_3

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

def train(model, train_inputs, train_labels):
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # bacth number
    batch_number = (np.ceil(np.divide(len(train_labels), model.batch_size))).astype(int)
    range_tensor = tf.constant(np.arange(train_inputs.shape[0]))
    # train in one loop
    for j in range(10):
        indices = tf.random.shuffle(range_tensor)
        shuffled_inputs = tf.gather(train_inputs,indices)
        shuffled_labels = tf.gather(train_labels,indices)
        for i in range(batch_number):
            if i < batch_number - 1:
                X = shuffled_inputs[i * model.batch_size :(i+1) * model.batch_size]
                Y = shuffled_labels[i * model.batch_size :(i+1) * model.batch_size]
            else:
                X = shuffled_inputs[i * model.batch_size :len(train_labels)]
                Y = shuffled_labels[i * model.batch_size :len(train_labels)]
        # Implement backprop:
            with tf.GradientTape() as tape:
                predictions = model.call(X) # this calls the call function conveniently
                losses = model.loss(predictions,Y)
      
            #if i % 1000 == 0:
               #train_acc = model.accuracy(model.call(train_inputs), train_labels)
               #print("Accuracy on training set after {} training steps: {}".format(i, train_acc))
    
        # The keras Model class has the computed property trainable_variables to conveniently
        # return all the trainable variables you'd want to adjust based on the gradients
            gradients = tape.gradient(losses, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    pass

def test(model, test_inputs, test_labels):
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

    logits = model.call(test_inputs,is_testing = True)
    accuracy = model.accuracy(logits,test_labels)
    print("test accuracy",accuracy)
    return accuracy


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


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
    train_inputs_file_path = 'data/train'
    test_inputs_file_path = 'data/test'
    train_inputs, train_labels = get_data(train_inputs_file_path,3,5)
    test_inputs, test_labels = get_data(test_inputs_file_path, 3,5)
    model = Model()
    train(model,train_inputs,train_labels)
    test(model,test_inputs,test_labels)
    
    return


if __name__ == '__main__':
    main()
