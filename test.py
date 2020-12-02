import tensorflow as tf
import numpy as np
from preprocess import get_data
from myplain16 import Plain16
from ResBlock import ResBlock
from resnet import ResNet18
from resnet import ResNet32
from matplotlib import pyplot as plt

import random

def visualize_loss_accuracy(losses,accuracy): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x1 = [i for i in range(len(losses))]
    plot1 = plt.figure(1)
    plt.plot(x1, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    x2 = [i for i in range(len(accuracy))]
    plot2 = plt.figure(2)
    plt.plot(x2, accuracy)
    plt.title('Accuracy per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
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
            model.loss_list.append(losses.numpy())
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
    model = ResNet32()
    train_images, train_labels, test_images, test_labels = get_data()
    accuracy_list =[]
    num_epochs = 20
    for i in range(num_epochs):
        print(i)
        train(model,train_images,train_labels)
        test_accuracy = test(model,test_images,test_labels)
        accuracy_list.append(test_accuracy)
        print(test_accuracy)
    visualize_loss_accuracy(model.loss_list,accuracy_list)
if __name__ == '__main__':
    main()
