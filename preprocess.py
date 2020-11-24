import pickle
import numpy as np
<<<<<<< Updated upstream
import tensorflow as tf
import os
=======
import tensorflow_datasets as tfds

"""
This module is responsible for the preprocessing of the image datasets
"""

"""
This function loads the CIFAR10 datasets
return:
train_images - numpy array of the shape (50000,32,32,3)
train_labels - numpy array of the shape (50000,)
test_images - numpy array of the shape (10000,32,32,3)
test_labels - numpy array of the shape (10000,)
"""
def get_data():
    data = tfds.load("cifar10")
    train, test = data["train"],data["test"]

    train_iter = train.as_numpy_iterator()
    test_iter = test.as_numpy_iterator()
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for datum in train_iter:
        train_images.append(datum["image"])
        train_labels.append(datum["label"])
    for datum in test_iter:
        test_images.append(datum["image"])
        test_labels.append(datum["label"])
    norm_factor = 1/255
    train_images = np.array(train_images,dtype=np.float32)*norm_factor
    train_images = train_images
    train_labels = np.array(train_labels)
    test_images = np.array(test_images,dtype=np.float32)*norm_factor
    test_images = test_images
    test_labels = np.array(test_labels)
>>>>>>> Stashed changes

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
      
    
    
    