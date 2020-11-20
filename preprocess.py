import pickle
import numpy as np
import tensorflow as tf
import os

def get_data(file_path, num_classes, classes):
	#Given a file path and num_classes, classes which is a list of  indexes indicating the classes 
    #that we want to classify, returns an array of 
	#normalized inputs (images) and an array of labels. 
	#You will want to first extract only the data that matches the 
	#corresponding classes we want (there are 10 classes in CIFAR10).
	#You should make sure to normalize all inputs and also turn the labels
	#into one hot vectors using tf.one_hot().
	#Note that because you are using tf.one_hot() for your labels, your
	#labels will be a Tensor, while your inputs will be a NumPy array. This 
	#is fine because TensorFlow works with NumPy arrays.
	#:param file_path: file path for inputs and labels, something 
	#like 'CIFAR_data_compressed/train'
    #:param num_classes: number of classes that we want to train on
	#:param classes:  a list of integers  (0-9) and the length is  num_classes
	#:return: normalized NumPy array of inputs and tensor of labels, where 
	#inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels 
	#has size (num_examples, num_classes)
    return inputs, labels
      
    
    
    