import argparse


def parse_input(arg, autoencoder=True):
    """ function used to parse the command line input of the autoencoder """

    # create the argument parser
    description = "Python script that creates an autoencoder used to reduce the dimensionality of "
                  "the MNIST dataset."
    parser = argparse.ArgumentParser(description=description)

    # add an argument for the path of the dataset
    help = "The full/relative path to the file containing the training examples."
    parser.add_argument("-d", "--data", type=str, action="store", metavar="training_examples_path",
                        required=True, help=help)

    # if this is the input for the autoencoder, then we are ok; else parse the rest arguments
    if not autoencoder:

        # add an argument for the path of the training labels
        help = "The full/relative path to the file containing the training labels."
        parser.add_argument("-dl", "--datalabels", type=str, action="store",
                            metavar="training_labels_path", required=True, help=help)

        # add an argument for the path of the testing examples
        help = "The full/relative path to the file containing the testing examples."
        parser.add_argument("-t", "--test", type=str, action="store",
                            metavar="testing_examples_path", required=True, help=help)

        # add an argument for the path of the testing labels
        help = "The full/relative path to the file containing the testing labels."
        parser.add_argument("-tl", "--testlabels", type=str, action="store",
                            metavar="testing_labels_path", required=True, help=help)

    # parse the arguments and return the result
    return parser.parse_args(arg)
