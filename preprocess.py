import tensorflow as tf
import numpy as np
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
    train_labels = np.array(train_labels)
    test_images = np.array(test_images,dtype=np.float32)*norm_factor
    test_labels = np.array(test_labels)

    return train_images, train_labels, test_images, test_labels

