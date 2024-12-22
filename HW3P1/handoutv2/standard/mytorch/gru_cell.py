import numpy as np
from mytorch.nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        # given current observation and previous hidden calculate hidden at current

        # how current observation affect current hidden
        # Wrx :  (h, d)
        self.r = self.r_act.forward(np.dot(self.Wrx, self.x) + self.brx + np.dot(self.Wrh, self.hidden) + self.brh)
        self.z = self.z_act.forward(np.dot(self.Wzx, self.x) + self.bzx + np.dot(self.Wzh, self.hidden) + self.bzh)
        self.reset_val = np.dot(self.Wnh, self.hidden) + self.bnh
        self.n = self.h_act.forward(np.dot(self.Wnx, self.x) + self.bnx + (self.r * self.reset_val))
        h_t = (1-self.z) * self.n  + self.z * self.hidden

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)  # h_t is the final output of you GRU cell.

        return h_t



    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        delta = delta.reshape(-1, 1)
        x = self.x.reshape(-1, 1)
        h_prev = self.hidden.reshape(-1, 1)  
        r_t = self.r.reshape(-1, 1)
        z_t = self.z.reshape(-1, 1)
        n_t = self.n.reshape(-1, 1)


        # update gate
        dz = delta * (h_prev - n_t) *  z_t * (1 - z_t)
        # candidate hidden
        dn = delta * (1 - self.z).reshape(-1, 1) * (1 - n_t ** 2)
        # retrieve saved rest_val in forward pass
        rest_val = self.reset_val.reshape(-1,1)
        # reset 
        dr = (dn * rest_val) * r_t * (1 - r_t)
        dx = np.dot(self.Wnx.T, dn) + np.dot(self.Wrx.T, dr) + np.dot(self.Wzx.T, dz)
        

        dh_prev_t = delta * z_t + (self.Wrh.T @ dr) + (self.Wzh.T @ dz) + (self.Wnh.T @ (dn * r_t))

    
        self.dWnx += np.dot(dn, x.T)
        self.dWnh += np.dot(dn * r_t, h_prev.T)
        self.dbnx += dn.reshape(-1)
        self.dbnh += (dn * r_t).reshape(-1)


        self.dWrx += np.dot(dr, x.T)
        self.dWrh += np.dot(dr, h_prev.T)
        self.dbrx += dr.reshape(-1)
        self.dbrh += dr.reshape(-1)

        self.dWzx += np.dot(dz, x.T)
        self.dWzh += np.dot(dz, h_prev.T)
        self.dbzx += dz.reshape(-1)
        self.dbzh += dz.reshape(-1)

        dx = dx.reshape(self.d)
        dh_prev_t = dh_prev_t.reshape(self.h)


        return dx, dh_prev_t
     