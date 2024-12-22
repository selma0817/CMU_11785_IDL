# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
from typing import Any
import numpy as np

class Sigmoid:
    """
    Sigmoid activation function
    """

    def forward(self, Z):

        self.A = Z
        self.sigmoid_val = 1 / (1 + np.exp(-self.A))
        return self.sigmoid_val

    def backward(self, dLdA):

        dAdZ = self.sigmoid_val * (1 - self.sigmoid_val)
        return dAdZ * dLdA

class Tanh:
    """
    Modified Tanh to work with BPTT.
    The tanh(x) result has to be stored elsewhere otherwise we will
    have to store results for multiple timesteps in this class for each cell,
    which could be considered bad design.

    Now in the derivative case, we can pass in the stored hidden state and
    compute the derivative for that state instead of the "current" stored state
    which could be anything.
    """
    def forward(self, Z):

        self.A = Z
        self.tanhVal =  np.tanh(self.A)
        return self.tanhVal

    def backward(self, dLdA, state=None):
        if state is not None:
            # dAdZ = 1 - state*state
            # return dAdZ * dLdA
            return 1 - (state*state)
        else:
            # dAdZ = 1 - self.tanhVal * self.tanhVal
            # return dAdZ * dLdA
            return 1 - (self.tanhVal * self.tanhVal)
