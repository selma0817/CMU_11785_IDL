import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        (batch_size, in_channels, input_width) = A.shape
        k = self.upsampling_factor
        output_width = k * (input_width - 1) + 1
        Z = np.zeros((batch_size, in_channels, output_width))
        Z[:, :, ::k] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        (batch_size, in_channels, output_width) = dLdZ.shape
        k = self.upsampling_factor
        input_width = (output_width - 1) // k + 1
        positions = np.arange(input_width) * k

        dLdA = dLdZ[:, :, positions]  

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.input_width = None
    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        (batch_size, in_channels, input_width) = A.shape
        self.input_width = input_width
        k = self.downsampling_factor

        Z = A[:, :, ::k] # step of k
    
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
               
        (batch_size, in_channels, output_width) = dLdZ.shape
        k = self.downsampling_factor
        
        dLdA = np.zeros((batch_size, in_channels, self.input_width))
        
        dLdA[:, :, ::k] = dLdZ


        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        (batch_size, in_channels, input_height, input_width) = A.shape
        k = self.upsampling_factor
        output_width = k * (input_width - 1) + 1
        output_height = k * (input_height - 1) + 1
        Z = np.zeros((batch_size, in_channels, output_height, output_width))
        Z[:, :, ::k, ::k] = A

        return Z 

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
   
        k = self.upsampling_factor

        
        dLdA = dLdZ[:, :, ::k, ::k]  # TODO

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.input_height = None
        self.input_width = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        
        (batch_size, in_channels, input_height, input_width) = A.shape
        self.input_height = input_height
        self.input_width = input_width
        k = self.downsampling_factor
        Z = A[:, :, ::k, ::k]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        (batch_size, in_channels, output_height, output_width) = dLdZ.shape
        k = self.downsampling_factor

        dLdA = np.zeros((batch_size, in_channels, self.input_height, self.input_width))

        dLdA[:, :, ::k, ::k] = dLdZ
        return dLdA
