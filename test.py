import tensorflow as tf
import numpy as np
from preprocess import get_data
from myplain16 import Plain16
from myplain16 import Model

import random

def train(model, train_inputs, train_labels):
<<<<<<< Updated upstream
    for j in range(5):
        print("j=",j)
        indices = tf.range(len(train_inputs))
        indices = tf.random.shuffle(indices)
        inputs = tf.gather(train_inputs, indices)
        labels = tf.gather(train_labels, indices)
        for i in range(0,len(train_inputs),model.batch_size):
            print(i/model.batch_size)
            batch_inputs = inputs[i:i+model.batch_size]
            batch_labels = labels[i:i+model.batch_size]
            batch_inputs = tf.image.random_flip_left_right(batch_inputs)
            with tf.GradientTape() as tape:
                logits = model.call(batch_inputs)
                losses = model.loss(logits,batch_labels)
            print("j=, i=,losses =",j, i,losses)
            gradients = tape.gradient(losses,model.trainable_variables)
            #print("gradients",gradients)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return
=======
    indices = tf.range(len(train_inputs))
    indices = tf.random.shuffle(indices)
    inputs = tf.gather(train_inputs, indices)
    labels = tf.gather(train_labels, indices)
    dataset = tf.data.Dataset.from_tensor_slices((inputs,labels)).batch(model.batch_size,drop_remainder=True)
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
>>>>>>> Stashed changes
def test(model, test_inputs, test_labels):
    accum = 0
    for i in range(0,len(test_inputs)-model.batch_size,model.batch_size):
        batch_inputs = test_inputs[i:i+model.batch_size]
        batch_labels = test_labels[i:i+model.batch_size]
<<<<<<< Updated upstream
        logits = model.call(batch_inputs)
        acc = model.accuracy(logits,batch_labels).numpy()
=======
        probs = model.call(batch_inputs,is_testing=True)
        acc = model.accuracy(probs,batch_labels).numpy().item()
        print("accuaracy=",acc)
>>>>>>> Stashed changes
        accum+=acc
    return accum/(np.floor(len(test_inputs)/model.batch_size))

def main():
    model = Plain16()
    train_images, train_labels, test_images, test_labels = get_data()
<<<<<<< Updated upstream
    train(model,train_images,tf.one_hot(train_labels,10))
    print(test(model,test_images,tf.one_hot(test_labels,10)))
=======
    for i in range(25):
        print(i)
        train(model,train_images,train_labels)
        print(test(model,test_images,test_labels))
>>>>>>> Stashed changes



if __name__ == '__main__':
    main()
