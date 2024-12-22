# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')


class CNN_SimpleScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        # in_channel same as input dim, out channel is num of neuron in this layer

        self.conv1 = Conv1d(in_channels= 24, out_channels= 8, kernel_size = 8, stride=4)
        # we want to preserve having 31 output for each channel, so we need kernel size of 1 and stride of 1
        self.conv2 = Conv1d(in_channels= 8, out_channels= 16, kernel_size = 1, stride=1)
        self.conv3 = Conv1d(in_channels= 16, out_channels= 4, kernel_size = 1, stride=1)
        # by doing so we get 4 * 31 at the end as required  
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(),self.conv3, Flatten()]
        
    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        # print(f"w1 shape: {w1.shape}")
        # print(f"w2 shape: {w2.shape}")
        # print(f"w3 shape: {w3.shape}")
        # w1 shape: (192, 8) # (input_neuron, output_neuron)
        # w2 shape: (8, 16)
        # w3 shape: (16, 4)

        # weight for conv1d_stride1 is shape (out_channels, in_channels, kernel_size)
        # first transpose to (output_neuron, input_neuron)
        self.conv1.conv1d_stride1.W = w1.T.reshape(8, 8, 24).transpose(0, 2, 1)
        self.conv2.conv1d_stride1.W = w2.T.reshape(16, 1, 8).transpose(0, 2, 1)
        self.conv3.conv1d_stride1.W = w3.T.reshape(4, 1, 16).transpose(0, 2, 1)


    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(in_channels= 24, out_channels= 2, kernel_size = 2, stride=2)
        self.conv2 = Conv1d(in_channels= 2, out_channels= 8, kernel_size = 2, stride=2)
        self.conv3 = Conv1d(in_channels= 8, out_channels= 4, kernel_size = 2, stride=1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(),self.conv3, Flatten()]
        
    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        # mlp weight is : (output_neurons, input_neurons)
        # conv layer weight is: (out_channels, in_channels, kernel_size)

        w1, w2, w3 = weights
        # print(f"w1 shape: {w1.shape}")
        # print(f"w2 shape: {w2.shape}")
        # print(f"w3 shape: {w3.shape}")
        # w1 shape: (192, 8) in neuron, out neuron
        # w2 shape: (8, 16)
        # w3 shape: (16, 4)

        # w1 -> (2, 24, 2) # (out_channel, in_channel, kernel)
        # (in neuron, out neuron) -> (out neuron , in neuron) -> (out, kernel, in) -> (out, in, kernel)
        # 24*2 = 48
        self.conv1.conv1d_stride1.W = w1[:48, :2].T.reshape(2,2,24).transpose(0,2,1)
        # w2 -> (8, 2, 2) 2*2 = 4
        self.conv2.conv1d_stride1.W = w2[:4, :8].T.reshape(8,2,2).transpose(0,2,1)
        # w3 -> (4, 8, 2)
        # 2*8 = 16, no need slicing 
        self.conv3.conv1d_stride1.W = w3.T.reshape(4,2,8).transpose(0,2,1)

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
