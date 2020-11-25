import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, LeakyReLU
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.activations import relu, softmax


def create_classifier(rows, columns, encoder, units):
    """
    Function that given the encoder part and the dense part of a classifier, creates a "Model"
    (Keras object) that represents a classifier.
    """

    # define the input
    input = Input(shape=(rows, columns, 1))
    x = input

    # pass the input through the encoder
    x = encoder(input)(x)

    x = Flatten()(x)
    # pass then the result through the dense layer
    x = Dense(units=units, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)

    # create the model and return it
    classifier = Model(input, x, name="classifier")
    return classifier


def show_experiment_graph(history):
    """ Function used to show the Loss vs Epochs graph of one experiment """

    # get the losses
    train_losses = history.history["mse"]
    val_losses = history.history["val_mse"]

    # plot the losses
    epochs = len(train_losses)
    plt.xticks(np.arange(0, epochs, 1), np.arange(1, epochs + 1, 1))
    plt.plot(train_losses, label="Train Loss", color="mediumblue")
    plt.plot(val_losses, label="Validation Loss", color="darkred")

    # define some more parameters
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def show_graphs(histories, configurations):
    """ Function used to plot the losses of a model, for each configuration (experiment) tried """

    # get the number of experiments performed
    experiments = len(histories)

    # get the last losses of each experiment
    train_losses = [history.history["mse"][-1] for history in histories]
    val_losses = [history.history["val_mse"][-1] for history in histories]

    # now fix the x labels to match every experiment
    xlabels = []
    # add a label for each configuration
    for configuration in configurations:
        # get the values for that configuration
        units, epochs, batch_size = configuration
        # define the string and append it
        xlabel = "Units: {}\nEpochs: {}\nBatch Size: {}\n".format(units, epochs, batch_size)
        xlabels.append(xlabel)

    # define the parameters of the plot
    plt.xticks(np.arange(experiments), xlabels)

    # plot the losses
    plt.plot(train_losses, label="Train Loss", color="mediumblue")
    plt.plot(val_losses, label="Validation Loss", color="darkred")

    # define some more parameters
    plt.xlabel("Runs")
    plt.ylabel("Losses")
    plt.legend()
    plt.show()
