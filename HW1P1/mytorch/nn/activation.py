import numpy as np
import math
from scipy.special import erf

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A
    
    def backward(self, dLdA):
        return dLdA * (self.A * (1 - self.A))


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):
        self.A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        return self.A
    def backward(self, dLdA):
        return dLdA * (1 - self.A**2)


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self, Z):
        self.A = np.maximum(0, Z)
        # not using np.max here, np max return single element max along an axis
        # np maximum compare every num in vector with 0 in this case
        return self.A
    def backward(self, dLdA):
        return dLdA * np.where(self.A > 0, 1, 0)

class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU.
    """
    def forward(self, Z):
         self.E = 0.5 * (1 + erf(Z / np.sqrt(2)))
         self.A = Z * self.E
         self.Zp = (Z / np.sqrt(2 * np.pi)) * np.exp(-0.5 * Z**2)
         return self.A
    def backward(self, dLdA):
        return dLdA * (self.E + self.Zp)


class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        exp_Z = np.exp(Z)
        row_sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
        ## need to keep dim so it doesn't become (N, )

        self.A = exp_Z / row_sum_exp_Z 
        self.N = Z.shape[0]
        self.C = Z.shape[1]
        return self.A
    
    def backward(self, dLdA):

        # Calculate the batch size and number of features
        N = self.N
        C = self.C

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros((N, C))

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros((C, C))

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    if(m == n):
                        J[m,n] = self.A[i, m] * (1 - self.A[i, m])  
                    else:
                        J[m,n] = -self.A[i, m] * self.A[i, n]


            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i,:] = dLdA[i, :] @ J

        return dLdZ