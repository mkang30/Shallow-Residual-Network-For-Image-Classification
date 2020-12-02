import tensorflow as tf
import numpy as np
from preprocess import get_data
from myplain16 import Plain16
from ResBlock import ResBlock
from resnet import ResNet18

import random

def train(model, train_inputs, train_labels):
    indices = tf.range(len(train_inputs))
    indices = tf.random.shuffle(indices)
    inputs = tf.gather(train_inputs, indices)
    labels = tf.gather(train_labels, indices)
    dataset = tf.data.Dataset.from_tensor_slices((inputs,labels))
    dataset = dataset.interleave(lambda x,y: tf.data.Dataset.from_tensors((x,y)), num_parallel_calls=3)
    dataset = dataset.batch(model.batch_size,drop_remainder=True)
    dataset = dataset.prefetch(1)

    for i,(batch_inputs,batch_labels) in enumerate(dataset):
        batch_inputs = tf.image.random_flip_left_right(batch_inputs)
        with tf.GradientTape() as tape:
            probs = model.call(batch_inputs)
            losses = model.loss(probs,batch_labels)
        if i%10 ==0:
            print("i={},loss={}".format(i,losses))
        gradients = tape.gradient(losses,model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
    accum = 0
    dataset = tf.data.Dataset.from_tensor_slices((test_inputs,test_labels)).batch(model.batch_size,drop_remainder=True)
    dataset = dataset.prefetch(1)
    for i,(batch_inputs,batch_labels) in enumerate(dataset):
        probs = model.call(batch_inputs,is_testing=True)
        acc = model.accuracy(probs,batch_labels).numpy().item()
        print("accuaracy=",acc)
        accum+=acc
    return accum/(np.floor(len(test_inputs)/model.batch_size))

def main():
    model = ResNet18()
    train_images, train_labels, test_images, test_labels = get_data()
    for i in range(25):
        print(i)
        train(model,train_images,train_labels)
        print(test(model,test_images,test_labels))

if __name__ == '__main__':
    main()
