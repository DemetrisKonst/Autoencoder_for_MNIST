import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LeakyReLU

import logging
from utils import *



def main(args):
    """ main() driver function """

    # first make sure that the path to the provided dataset is valid
    if filepath_is_not_valid(args.data):
        logging.error("The path {} is not a file. Aborting..".format(args.data))
        exit()

    # get the data from the training set
    X = parse_dataset(args.data)
    # y = parse_labelset("../Dataset/train-labels-idx1-ubyte")

    print_image(X[0], 28, 28)
    plot_image(X[0])
    # print(y[0])

    print_image(X[1], 28, 28)
    plot_image(X[1])
    # print(y[1])


if __name__ == "__main__":
    """ call main() function here """
    print()
    # configure the level of the logging and the format of the messages
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s\n")
    # parse the command line input
    args = parse_input(autoencoder=True)
    # call the main() driver function
    main(args)
