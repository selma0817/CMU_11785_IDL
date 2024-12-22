import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z
        self.N = Z.shape[0] # N is batch size, Z is of shape (N x C)
        self.C = Z.shape[1]
        self.M = np.zeros((1, self.C))  # TODO # M has shape (1, C). minibatch per feature min
        self.V = np.ones((1, self.C))

        if eval == False:
            # training mode
            self.M = 1/self.N * np.sum(self.Z, axis=0) # shape 1, C
            self.V = 1/self.N * np.sum((self.Z - self.M)**2, axis=0 ) # shape 1, C
            # np.sqrt(+ self.eps)
            STD0 = np.sqrt(self.V + self.eps)
            self.NZ = (self.Z -  self.M)  / STD0
            self.BZ = self.BW * self.NZ + self.Bb

            self.running_M = self.alpha * self.running_M + (1-self.alpha)*self.M
            self.running_V = self.alpha * self.running_V + (1-self.alpha)*self.V
            
            return self.BZ
        else:
            # inference mode
            NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            BZ = self.BW * NZ + self.Bb

        return BZ

    def backward(self, dLdBZ):

        self.dLdBW = np.sum(dLdBZ * self.NZ)
        self.dLdBb = np.sum(dLdBZ, axis=0)

        dLdNZ = dLdBZ * self.BW
        dLdV = -0.5 * np.sum(dLdNZ * (self.Z- self.M) * (self.V + self.eps )**(-1.5), axis = 0)
        
        dNZdM =  -(self.V + self.eps)**(-0.5) - 0.5*(self.Z-self.M) * (self.V + self.eps)**(-1.5)*((-2/self.N)*np.sum(self.Z - self.M, axis=0))
        dLdM = np.sum(dLdNZ * dNZdM ,axis=0)
        dLdZ = dLdNZ * ((self.V+self.eps)**(-0.5)) + dLdV * ((2/self.N)*(self.Z-self.M)) + (1/self.N)*dLdM

        return dLdZ
