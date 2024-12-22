import numpy as np
# # Copy your Linear class from HW1P1 here
# class MSELoss:

#     def forward(self, A, Y):

#         self.A = A
#         self.Y = Y
#         N = A.shape[0]
#         C = A.shape[1]
#         se = None  # TODO
#         sse = None  # TODO
#         mse = sse / (N * C)

#         return NotImplemented

#     def backward(self):

#         dLdA = None

#         return NotImplemented


# class CrossEntropyLoss:

#     def forward(self, A, Y):

#         self.A = A
#         self.Y = Y
#         N = A.shape[0]
#         C = A.shape[1]
#         Ones_C = np.ones((C, 1), dtype="f")
#         Ones_N = np.ones((N, 1), dtype="f")

#         self.softmax = None  # TODO
#         crossentropy = None  # TODO
#         sum_crossentropy = None  # TODO
#         L = sum_crossentropy / N

#         return NotImplemented

#     def backward(self):

#         dLdA = None  # TODO

#         return NotImplemented



class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]
        se = (A - Y) * (A - Y)
        sse = np.ones((1, self.N)) @ se @ np.ones((self.C, 1))
        mse = sse / (self.N * self.C)

        return mse

    def backward(self):

        dLdA = 2 * (self.A - self.Y) / (self.N * self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]

        Ones_C = np.ones((self.C, 1))
        Ones_N = np.ones((self.N, 1))

        self.softmax = np.exp(self.A) / np.sum(np.exp(A), axis=1, keepdims=True)
        crossentropy = (-self.Y * np.log(self.softmax)) @ Ones_C
        sum_crossentropy = Ones_N.T @ crossentropy
        L = sum_crossentropy / self.N

        return L

    def backward(self):

        dLdA = (self.softmax - self.Y) / self.N

        return dLdA
