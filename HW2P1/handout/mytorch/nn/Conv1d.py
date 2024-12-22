# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        batch_size, in_channels, input_size = A.shape
        output_size = input_size - self.kernel_size + 1 # our stride is 1 here so div by 1

        out = np.zeros((batch_size, self.out_channels, output_size)) # TODO
        # for bs in range(batch_size):
        #     for oc in range(self.out_channels):
        #         for w in range(output_size):
        #             Z[bs,oc,w] = np.sum(A[bs, :, w:w+self.kernel_size] * self.W[oc,:,:]) +self.b[oc]
        for o in range(output_size):
            x_filter = A[:, :, o :o + self.kernel_size]  
            out[:, :, o] = np.tensordot(x_filter, self.W, ([1, 2], [1, 2]))  # (batch,out_channel)
        out += np.reshape(self.b, (self.b.shape[0], 1))
        return out

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        
        
        batch_size, out_channels, output_size = dLdZ.shape
        batch_size, in_channels, input_size = self.A.shape
        
        # sum to find dLdb
        self.dLdb = np.sum(dLdZ, axis = (0,2))
    
        # dLdW of shape (out_channel, in_channel, kernel size)
        # for oc in range(self.out_channels):
        #     for ic in range(self.in_channels):
        #         for k in range(self.kernel_size):
        #             self.dLdW[oc,ic,k] += np.sum(self.A[:,ic,k:k+output_size]*dLdZ[:,oc,:])
                   
        for o in range(output_size):
            x_slice = self.A[:, :, o:o+self.kernel_size]
            self.dLdW += np.tensordot(dLdZ[:, :, o], x_slice, axes=(0, 0))
        
        dLdA = np.zeros(self.A.shape) 
        
        flipped_W = np.flip(self.W, axis=2)
        padz = np.pad(dLdZ, ((0,0),(0,0),(self.kernel_size - 1,self.kernel_size - 1)), 'constant') 
        # for bs in range(batch_size):
        #     for ic in range(in_channels):
        #         for w in range(input_size):
        #             dLdA[bs, ic, w] += np.sum(padz[bs,:,w:w+self.kernel_size]*flipped_W[:,ic,:])
        for i in range(input_size):
            dLdA[:, :, i] = np.tensordot(padz[:, :, i:i+self.kernel_size], flipped_W, axes=([1, 2], [0, 2]))
        return dLdA

 


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding
        
        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 =Conv1d_stride1(in_channels, out_channels, kernel_size,weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Pad the input appropriately using np.pad() function
        if self.pad > 0:
            A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), mode='constant')
        else:
            A_padded = A
        # Call Conv1d_stride1
        Z = self.conv1d_stride1.forward(A_padded)

        if self.stride > 1:
            Z = self.downsample1d.forward(Z)


        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        if self.stride > 1:
            dLdZ = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ)

        if self.pad > 0:
            dLdA = dLdA[:, :, self.pad:-self.pad]


        return dLdA
