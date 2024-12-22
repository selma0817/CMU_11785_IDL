import numpy as np

# # Copy your Linear class from HW1P1 here
# class Linear:

#     def __init__(self, in_features, out_features, debug=False):

#         self.W = np.zeros((out_features, in_features), dtype="f")
#         self.b = np.zeros((out_features, 1), dtype="f")
#         self.dLdW = np.zeros((out_features, in_features), dtype="f")
#         self.dLdb = np.zeros((out_features, 1), dtype="f")

#         self.debug = debug

#     def forward(self, A):

#         self.A = A
#         self.N = A.shape[0]
#         self.Ones = np.ones((self.N, 1), dtype="f")
#         Z = self.A @ self.W.T +  self.Ones @ self.b.T  # TODO

#         return Z

#     def backward(self, dLdZ):

#         dZdA = dLdZ @ self.W  # TODO
#         dZdW = None  # TODO
#         dZdi = None
#         dZdb = None  # TODO
#         dLdA = None  # TODO
#         dLdW = None  # TODO
#         dLdi = None
#         dLdb = None  # TODO
#         self.dLdW = dLdW / self.N
#         self.dLdb = dLdb / self.N

#         if self.debug:

#             self.dZdA = dZdA
#             self.dZdW = dZdW
#             self.dZdi = dZdi
#             self.dZdb = dZdb
#             self.dLdA = dLdA
#             self.dLdi = dLdi

#         return NotImplemented


class Linear:

    def __init__(self, in_features, out_features, debug=False, weight_init_fn=None, bias_init_fn=None):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        # self.W = np.zeros((out_features, in_features)) # TODO
        # self.b = np.zeros((out_features, 1))

        self.debug = debug

        if weight_init_fn is None:
            self.W = np.zeros((out_features, in_features))
        else:
            self.W = weight_init_fn(out_features, in_features)

        if bias_init_fn is None:
            self.b = np.zeros((out_features, 1))
        else:
            self.b = bias_init_fn(out_features)


    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A
        self.N = A.shape[0]  # TODO store the batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        self.Ones = np.ones((self.N,1))
        Z = self.A @ self.W.T +  self.Ones @ self.b.T # TODO

        return Z

    def backward(self, dLdZ):

        dLdA = dLdZ @ self.W
        self.dLdW = dLdZ.T @ self.A
        self.dLdb = dLdZ.T @ self.Ones

        if self.debug:
            
            self.dLdA = dLdA

        return dLdA

