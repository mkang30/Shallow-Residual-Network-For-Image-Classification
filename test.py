import tensorflow as tf
import numpy as np
from preprocess import get_data
from myplain16 import Plain16

import random

def train(model, train_inputs, train_labels):
    indices = tf.range(len(train_inputs))
    indices = tf.random.shuffle(indices)
    inputs = tf.gather(train_inputs, indices)
    labels = tf.gather(train_labels, indices)

    for i in range(0,len(train_inputs),model.batch_size):
        batch_inputs = inputs[i:i+model.batch_size]
        batch_labels = labels[i:i+model.batch_size]
        batch_inputs = tf.image.random_flip_left_right(batch_inputs)
        with tf.GradientTape() as tape:
            probs = model.call(batch_inputs)
            losses = model.loss(probs,batch_labels)
        gradients = tape.gradient(losses,model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
def test(model, test_inputs, test_labels):
    accum = 0
    for i in range(0,len(test_inputs),model.batch_size):
        batch_inputs = test_inputs[i:i+model.batch_size]
        batch_labels = test_labels[i:i+model.batch_size]
        probs = model.call(batch_inputs,is_testing=True)
        acc = model.accuracy(probs,batch_labels).numpy().item()
        accum+=acc
    return accum/(len(test_inputs)/model.batch_size)

def main():
    model = Plain16()
    train_images, train_labels, test_images, test_labels = get_data()
    for i in range(25):
        print(i)
        train(model,train_images,train_labels)
    print(test(model,test_images,test_labels))



if __name__ == '__main__':
    main()
