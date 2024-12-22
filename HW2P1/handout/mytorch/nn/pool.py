import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1
        out_channels = in_channels
        Z = np.zeros((batch_size, out_channels, output_width, output_height))
        # for bs in range(batch_size):
        #     for oc in range(out_channels):
        #         for i in range(output_width):
        #             for j in range(output_height):
        #                 Z[bs, oc, i, j] = np.max(A[bs, oc, i:i+self.kernel, j:j+self.kernel])

        for i in range(output_width):
            for j in range(output_height):
                window = A[:, :, i:i+self.kernel, j:j+self.kernel]
                Z[:, :, i, j] = np.max(window, axis=(2, 3)) # max over the kernal dim

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros(self.A.shape)
        batch_size, out_channels, output_width, output_height= dLdZ.shape
        # input_width = output_width - k + 1
        # output_width = input_width + k -1
        
        # for bs in range(batch_size):
        #     for c in range(out_channels):
        #         for i in range(output_width):
        #             for j in range(output_height):
        #                 # only the max val in kernel window has gradient, other don't 
        #                 window = self.A[bs, c, i:i+self.kernel, j:j+self.kernel]
        #                 max_val = np.max(window)
        #                 mask = (window == max_val)
        #                 num_max = np.sum(mask)
        #                 dLdA[bs, c, i:i+self.kernel, j:j+self.kernel] += (mask * dLdZ[bs, c, i, j]) / num_max
        
        for i in range(output_width):
            for j in range(output_height):
                window = self.A[:, :, i:i+self.kernel, j:j+self.kernel]
                max_vals = np.max(window, axis=(2, 3), keepdims = True)
                mask = (window==max_vals)
                dLdA[:, :, i:i+self.kernel, j:j+self.kernel] += mask * (dLdZ[:, :, i, j])[:, :, None, None] / np.sum(mask, axis=(2, 3), keepdims=True)

        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height= A.shape
        out_channels = in_channels
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1

        Z = np.zeros((batch_size, out_channels, output_width, output_height))
        # for bs in range(batch_size):
        #     for oc in range(out_channels):
        #         for i in range(output_width):
        #             for j in range(output_height):
        #                 Z[bs, oc, i, j] = np.mean(A[bs, oc, i:i+self.kernel, j:j+self.kernel])

        for i in range(output_width):
            for j in range(output_height):
                window = A[:, :, i:i+self.kernel, j:j+self.kernel]
                Z[:, :, i, j] = np.mean(window, axis=(2, 3))
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = np.zeros(self.A.shape)
        batch_size, out_channels, output_width, output_height = dLdZ.shape
        gradient = dLdZ / (self.kernel * self.kernel)
        # for bs in range(batch_size):
        #     for c in range(out_channels):
        #         for i in range(output_width):
        #             for j in range(output_height):
        #                 dLdA[bs, c, i:i+self.kernel, j:j+self.kernel] += gradient[bs, c, i, j]
        for i in range(output_width):
            for j in range(output_height):
                dLdA[:, :, i:i+self.kernel, j:j+self.kernel] += gradient[:, :, i, j][:, :, None, None]
        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        c_A = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(c_A)
        return Z

    def backward(self, dLdZ):
        """clear
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        tempz = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(tempz)
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        temp = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(temp)

        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        temp = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(temp)
        return dLdA
