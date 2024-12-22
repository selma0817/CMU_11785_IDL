import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, in_channels, input_height, input_width = np.shape(A)
        output_width = input_width - self.kernel_size + 1
        output_height =  input_height - self.kernel_size + 1
        out = np.zeros((batch_size, self.out_channels, output_width, output_height))

        # for bs in range(batch_size):
        #     for oc in range(self.out_channels):
        #         for i in range(output_width):
        #             for j in range(output_height):
        #                 out[bs, oc, i, j] = np.sum(A[bs, :, i:i+self.kernel_size, j:j+self.kernel_size] * self.W[oc, :, :]) + self.b[oc]
        
        for i in range(output_height):
            for j in range(output_width):
                A_slice = A[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                out[:, :, i, j] = np.tensordot(A_slice, self.W, axes=([1, 2, 3], [1, 2, 3]))
        
        
        Z = out  
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        batch_size, out_channels, output_height, output_width = dLdZ.shape
        batch_size, in_channels, input_height, input_width = self.A.shape
        
        self.dLdb = np.sum(dLdZ, axis = (0, 2, 3))
        

        # out_channels, in_channels, kernel_size, kernel_size
        self.dLdW = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        # for oc in range(self.out_channels):
        #     for ic in range(self.in_channels):
        #         for k1 in range(self.kernel_size):
        #             for k2 in range(self.kernel_size):
        #                 self.dLdW[oc, ic, k1, k2] += np.sum(self.A[:,ic, k1:k1+output_height, k2:k2+output_width]*dLdZ[:, oc, :, :])
        
        for i in range(output_height):
            for j in range(output_width):
                A_slice = self.A[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                self.dLdW += np.tensordot(dLdZ[:, :, i, j], A_slice, axes=(0,0))
            
        dLdA = np.zeros_like(self.A)
        # weight horizontal flip then vertical flip
        flipped_W = np.flip(self.W, axis=(2, 3))
        # pad dLdZ with K-1 zeros 
        padz = np.pad(dLdZ, ((0,0),(0,0),(self.kernel_size - 1, self.kernel_size - 1), (self.kernel_size - 1, self.kernel_size - 1)), 'constant') 

        # fill in for dLdA

        # dLdA = np.zeros(self.A.shape)
        # for bs in range(batch_size):
        #     for ic in range(self.in_channels):
        #         for i in range(input_height):
        #             for j in range(input_width):
        #                 dLdA[bs, ic, i, j] += np.sum(padz[bs, :, i:i+self.kernel_size, j:j+self.kernel_size] * flipped_W[:,ic,:,:])

        for i in range(input_height):
            for j in range(input_width):
                dLdA[:, :, i, j] = np.tensordot(padz[:, :, i:i+self.kernel_size, j:j+self.kernel_size], flipped_W, axes=([1,2, 3], [0, 2, 3]))
         
  
        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        
        # Pad the input appropriately using np.pad() function
        # TODO
        if self.pad > 0:
            A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant')
        else:
            A_padded = A
        # call stride 1
        x = self.conv2d_stride1.forward(A_padded)
        
        # downsample
        if self.stride > 1:
            Z = self.downsample2d.forward(x)
        else: # add yy here
            Z = x
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        if self.stride > 1:
            dLdZ = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)

        # Unpad the gradient
        if self.pad > 0:
            dLdA = dLdA[:, :, self.pad:-self.pad, self.pad:-self.pad]

        return dLdA

if __name__ == "__main__":


    # Input Y (2 channels, 6 elements per channel)
    Y = np.array([[1, 0, 1, 0, 1], 
              [0, 1, 0, 1, 0]])

# Filters (kernels)
    W1 = np.array([[1, 2], 
               [2, 1]])

    W2 = np.array([[0, 1], 
               [1, 0]])

    W3 = np.array([[3, 2], 
               [1, 0]])

# Gradient of the loss with respect to Z (dZ)
    dZ = np.array([[1, 1, 1, 1],
               [2, 1, 2, 1],
               [1, 2, 1, 2]])
    weight_init_fn = np.stack([W1, W2, W3], axis=0)
# We will use the Conv2d class to test the backward and forward pass.
    conv_layer = Conv2d(in_channels=1, out_channels=3, kernel_size=2, stride=1, weight_init_fn=weight_init_fn)

# Forward pass
    output = conv_layer.forward(Y.reshape(1, 1, 2, 5))  # Reshape to match (batch_size, channels, height, width)

# Backward pass (using dZ from the question)
    dLdA = conv_layer.backward(dZ) 
    print("Gradient with respect to the input (dLdA) from the backward pass:")
    print(dLdA)
    



