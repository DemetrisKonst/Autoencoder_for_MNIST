import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import logging
import sys
sys.path.append("../utils")
sys.path.append("classification_utilities")

from utils import *
from classification_utils import *
from classification_interface_utils import *

def main(args):
    # first make sure that the path to the provided dataset is valid
    if filepath_is_not_valid(args.data):
        logging.error("The path {} is not a file. Aborting..".format(args.data))
        exit()

    # get the data from the training set
    X = parse_dataset(args.data)
    rows = X.shape[1]
    columns = X.shape[2]

    # first make sure that the path to the provided dataset is valid
    if filepath_is_not_valid(args.datalabels):
        logging.error("The path {} is not a file. Aborting..".format(args.datalabels))
        exit()

    # get the data from the training set
    Y = parse_labelset(args.datalabels)

    lb = LabelBinarizer()

    Y = lb.fit_transform(Y)

    # first make sure that the path to the provided dataset is valid
    if filepath_is_not_valid(args.test):
        logging.error("The path {} is not a file. Aborting..".format(args.test))
        exit()

    # get the data from the training set
    X_test = parse_dataset(args.test)

    # first make sure that the path to the provided dataset is valid
    if filepath_is_not_valid(args.testlabels):
        logging.error("The path {} is not a file. Aborting..".format(args.testlabels))
        exit()

    # get the data from the training set
    Y_test = parse_labelset(args.testlabels)

    Y_test = lb.transform(Y_test)

    # reshape so that the shapes are (number_of_images, rows, columns, 1)
    X = X.reshape(-1, rows, columns, 1)
    X_test = X_test.reshape(-1, rows, columns, 1)
    # normalize
    X = X / 255.
    X_test = X_test / 255.

    # split data to training and validation
    PELATES = 13
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=PELATES, shuffle=True)


    # use the list below to keep track of the configurations used for every experiment
    configurations = []
    # use the list below to keep track of the histories returned from each experiment
    histories = []

    # the option provided by the user
    option = 1

    # while the user wants to keep repeating the experiment
    while option != 0:
        configuration = get_classification_input()
        configurations.append(configuration)
        units, epochs, batch_size = configuration


        encoder = load_keras_model(args.modelpath)
        encoder.trainable = False

        classifier = create_classifier(rows, columns, encoder, units)
        print()
        classifier.summary()

        callback = ReduceLROnPlateau(monitor="val_loss", factor=1.0/2, patience=4, min_delta=0.005,
                                      cooldown=0, min_lr=1e-8, verbose=1)
        # callback = EarlyStopping(monitor="val_loss")

        classifier.compile(optimizer=optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["categorical_crossentropy", "accuracy"])

        history = classifier.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                                  shuffle=True, validation_data=(X_val, Y_val),
                                  callbacks=[callback])

        encoder.trainable = True

        print()
        classifier.summary()

        history = classifier.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                                  shuffle=True, validation_data=(X_val, Y_val),
                                  callbacks=[callback])

        # save this history
        histories.append(history)

        # get the new option from the user
        option = get_option()

        # while the user wants to either print the graphs or save the model, keep asking him
        while option == 2 or option == 3:

            # distinguish which option the user chose
            if option == 2:
                # get the next option on which graph to show
                answer = get_graph_option()

                # distinguish the answer
                if answer == 1:
                    # show the graph for the current experiment
                    show_experiment_graph(history)
                else:
                    # call the appropriate function to show the graphs of losses
                    show_graphs(histories, configurations)

            else:
                show_results(classifier, X_test, Y_test)

            # get the new option from the user
            option = get_option()

if __name__ == "__main__":
    """ call main() function here """
    print()
    # configure the level of the logging and the format of the messages
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s\n")
    # parse the command line input
    args = parse_input(autoencoder=False)
    # call the main() driver function
    main(args)
    print("\n")
