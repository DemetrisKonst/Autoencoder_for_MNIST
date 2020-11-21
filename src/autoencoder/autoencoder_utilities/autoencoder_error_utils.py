class UnequalLenError(Exception):
    """ Custom Error raised when the tokens of the an input do not match the expected length """

    def __int__(self, *args):
        """ constructor """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """ print when raised outside try block """
        message = "The amount of {} passed is not equal to the number of Convolutional Layers."
        if self.message:
            print(message.format(self.message))
        else:
            print(message.format("tokens"))


class InvalidTupleError(Exception):
    """ Custom Error raised when a tuple provided by the user is not of length 2 """

    def __int__(self, *args):
        """ constructor """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """ print when raised outside try block """
        message = "The tuple {}is not of length 2."
        if self.message:
            print(message.format(self.message + " "))
        else:
            print(message.format(""))
