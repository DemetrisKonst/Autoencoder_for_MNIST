import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

import logging
import sys
sys.path.append("../utils")
sys.path.append("autoencoder_utilities")

from utils import *
from autoencoder_utils import create_encoder, create_decoder, create_autoencoder
from autoencoder_interface_utils import get_autoencoder_input



def main(args):
    """ main() driver function """

    # first make sure that the path to the provided dataset is valid
    if filepath_is_not_valid(args.data):
        logging.error("The path {} is not a file. Aborting..".format(args.data))
        exit()

    # get the data from the training set
    X = parse_dataset(args.data)
    rows = X.shape[1]
    columns = X.shape[2]

    # reshape so that the shapes are (number_of_images, rows, columns, 1)
    X = X.reshape(-1, rows, columns, 1)
    # normalize
    X = X / 255.

    # split data to training and validation
    PANATHA = 13
    X_train, X_val = train_test_split(X, test_size=0.15, random_state=PANATHA, shuffle=True)

    # can be run with: python3 autoencoder.py -d ../../Dataset/train-images-idx3-ubyte < input.txt
    conv_layers, kernel_sizes, filters, epochs, batch_size, third_maxpool = get_autoencoder_input()

    # get the encoder and the decoder
    encoder = create_encoder(rows, columns, conv_layers, kernel_sizes, filters,
                             use_third_max_pooling=third_maxpool)
    decoder = create_decoder(rows, columns, conv_layers, kernel_sizes, filters,
                             use_third_max_pooling=third_maxpool)

    # get the autoencoder
    autoencoder = create_autoencoder(rows, columns, encoder, decoder)
    autoencoder.summary()

    # add a callback to reduce learning rate when validation loss plateaus
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=1.0/2, patience=4, min_delta=0.005,
                                  cooldown=0, min_lr=1e-8, verbose=1)

    # compile and train the autoencoder
    autoencoder.compile(optimizer=optimizers.Adam(1e-3), loss="mse", metrics=["mse"])
    history = autoencoder.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                              validation_data=(X_val, X_val), callbacks=[reduce_lr])


if __name__ == "__main__":
    """ call main() function here """
    print()
    # configure the level of the logging and the format of the messages
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s\n")
    # parse the command line input
    args = parse_input(autoencoder=True)
    # call the main() driver function
    main(args)
