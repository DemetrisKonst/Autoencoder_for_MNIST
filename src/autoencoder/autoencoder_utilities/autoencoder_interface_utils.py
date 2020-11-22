from autoencoder_error_utils import *

DEFAULT_CONV_LAYERS = 3
DEFAULT_KERNEL_SIZE = (3, 3)
DEFAULT_FILTERS = 16
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_USE_THIRD_MAX_POOL = True


def get_number_of_convolutional_layers():
    """ function used to read the number of convolutional layers """

    # get the number of convolutional layers
    prompt = "\nGive the number of Convolutional Layers for the Autoencoder (default = {}): "
    prompt = prompt.format(DEFAULT_CONV_LAYERS)
    conv_layers = input(prompt)

    # make sure the user gives a legit input
    while conv_layers != "":

        # try to convert the input to an int
        try:
            conv_layers = int(conv_layers)
            # it must be a positive integer
            if conv_layers <= 0:
                raise ValueError
            # if we get here then the input is fine, so break
            break

        # catch error and try again
        except ValueError:
            print("The number of Convolutional Layers must a positive integer. Please try again.")
            conv_layers = input(prompt)

    # check if the user wants to use the deault value
    if conv_layers == "":
        conv_layers = DEFAULT_CONV_LAYERS

    # return the final value
    return conv_layers


def get_kernel_sizes(conv_layers=DEFAULT_CONV_LAYERS):
    """ function used to read the kernel size for each convolutional layer """

    # read the kernel sizes
    prompt = "\nGive the kernel sizes (default = {} for all) for each Convolutional Layer as " \
             "a list (e.g.: (3, 3) (5, 5) (7, 7) ..): "
    prompt = prompt.format(DEFAULT_KERNEL_SIZE)
    kernel_sizes = input(prompt)

    # make sure that the input is legit
    while kernel_sizes != "":

        # get the input split
        split_kernel_sizes = kernel_sizes.split(")")
        # remove extra white spaces that may have been introduced by extra white space
        split_kernel_sizes = list(filter(lambda x: x != "", split_kernel_sizes))

        # try to convert the input to tuples of integers
        try:
            # check the restrictions
            if len(split_kernel_sizes) == conv_layers:

                # remove "(" character and white space
                split_kernel_sizes = list(map(lambda x: x.replace("(", ""), split_kernel_sizes))
                split_kernel_sizes = [kernel_size.strip() for kernel_size in split_kernel_sizes]

                # split in commas ","
                split_kernel_sizes = list(map(lambda x: x.split(","), split_kernel_sizes))

                # make sure that all tuples are of length 2
                for kernel_size in split_kernel_sizes:
                    if len(kernel_size) != 2:
                        raise InvalidTupleError

                # try to convert to integers
                split_kernel_sizes = [list(map(int, kernel)) for kernel in split_kernel_sizes]

                # try to convert to tuples
                kernel_sizes = [tuple(kernel_size) for kernel_size in split_kernel_sizes]

                # break if it succeeds
                break
            # else raise a value error as the number of kernel sizes provided was not correct
            else:
                raise UnequalLenError

        # else catch the value error and try again
        except ValueError:
            print("The kernel sizes must be tuples of positive integers. Please try again.")
            kernel_sizes = input(prompt)

        # catch wrong number of kernel sizes error and try again
        except UnequalLenError:
            kernels = len(split_kernel_sizes)
            print("The amount of kernel sizes passed ({}) is not equal to the number of " \
                  "Convolutional Layers ({}). Please try again.".format(kernels, conv_layers))
            kernel_sizes = input(prompt)

        # catch tuples with more than 2 components error and try again
        except InvalidTupleError:
            print("The kernel sizes have to be tuples of size exactly 2. Please try again.")
            kernel_sizes = input(prompt)

    # check if the user wants to use the default values
    if kernel_sizes == "":
        kernel_sizes = [DEFAULT_KERNEL_SIZE for kernel_size in range(conv_layers)]

    # return the final list
    return kernel_sizes


def get_filters(conv_layers=DEFAULT_CONV_LAYERS):
    """ function used to read the number of filters per convolutional layer """

    # read the filters
    prompt = "\nGive the filters (default = {} for all) for each Convolutional Layer as " \
             "a list (e.g.: 16 32 64 ..): "
    prompt = prompt.format(DEFAULT_FILTERS)
    filters = input(prompt)

    # make sure that the input is legit
    while filters != "":

        # get the input split
        split_filters = filters.split()

        # try to convert the input to integers
        try:
            # check the restrictions
            if len(split_filters) == conv_layers:
                # try to do the conversion
                filters = [int(filter) for filter in split_filters]
                # break if it succeeds
                break
            # else raise a value error as the number of kernel sizes provided was not correct
            else:
                raise UnequalLenError

        # else catch the value error and try again
        except ValueError:
            print("The filters must be positive integers. Please try again.")
            filters = input(prompt)

        # catch wrong number of filters error and try again
        except UnequalLenError:
            total_filters = len(split_filters)
            print("The amount of filters passed ({}) is not equal to the number of Convolutional " \
                  "Layers ({}). Please try again.".format(total_filters, conv_layers))
            filters = input(prompt)

    # check if the user wants to use the default values
    if filters == "":
        filters = [DEFAULT_FILTERS for filter in range(conv_layers)]

    # return the final list
    return filters


def get_epochs():
    """ function used to read the number of epochs perfomed in the training of the autoencoder """

    # get the number of epochs
    prompt = "\nGive the number of epochs (default = {}): "
    prompt = prompt.format(DEFAULT_EPOCHS)
    epochs = input(prompt)

    # make sure the user gives a legit input
    while epochs != "":

        # try to convert the input to an int
        try:
            epochs = int(epochs)
            # it must be a positive integer
            if epochs <= 0:
                raise ValueError
            # if we get here then the input is fine, so break
            break

        # catch error and try again
        except ValueError:
            print("The number of epochs must a positive integer. Please try again.")
            epochs = input(prompt)

    # check if the user wants to use the deault value
    if epochs == "":
        epochs = DEFAULT_EPOCHS

    # return the final value
    return epochs


def get_batch_size():
    """ function used to read the batch size used in the training of the autoencoder """

    # get the number of batch_size
    prompt = "\nGive the batch size (default = {}): "
    prompt = prompt.format(DEFAULT_BATCH_SIZE)
    batch_size = input(prompt)

    # make sure the user gives a legit input
    while batch_size != "":

        # try to convert the input to an int
        try:
            batch_size = int(batch_size)
            # it must be a positive integer
            if batch_size <= 0:
                raise ValueError
            # if we get here then the input is fine, so break
            break

        # catch error and try again
        except ValueError:
            print("The batch size must a positive integer. Please try again.")
            batch_size = input(prompt)

    # check if the user wants to use the deault value
    if batch_size == "":
        batch_size = DEFAULT_BATCH_SIZE

    # return the final value
    return batch_size


def get_usage_of_third_maxpool():
    """ function used to ask the user if he wants to use a third Max Pooling layer """

    # get the answer
    prompt = "\nWould you like to use a 3rd Max Pooling layer with pool size (7, 7) in the " \
             "architecture of the Neural Network? (y/n) (default = yes): "
    use_third_maxpool = input(prompt)

    # keep asking until he gives a valid answer
    while use_third_maxpool.lower() != "y" and use_third_maxpool.lower() != "yes" and \
          use_third_maxpool.lower() != "n" and use_third_maxpool.lower() != "no" and \
          use_third_maxpool != "":

        # print according message and read again input
        print("Invalid answer, please type (y/n).")
        use_third_maxpool = input(prompt)

    # check for default answer
    if use_third_maxpool == "":
        use_third_maxpool = DEFAULT_USE_THIRD_MAX_POOL
    else:
        use_third_maxpool = use_third_maxpool.lower() == "y" or use_third_maxpool.lower() == "yes"

    # return the final value
    return use_third_maxpool


def get_autoencoder_input():
    """ function used to get the input of the autoencoder """

    # get the input values one by one
    conv_layers = get_number_of_convolutional_layers()
    kernel_sizes = get_kernel_sizes(conv_layers)
    filters = get_filters(conv_layers)
    epochs = get_epochs()
    batch_size = get_batch_size()
    use_third_maxpool = get_usage_of_third_maxpool()

    # print some newlines and retun the values as a quintuple (5-tuple)
    print("\n")
    return conv_layers, kernel_sizes, filters, epochs, batch_size, use_third_maxpool
