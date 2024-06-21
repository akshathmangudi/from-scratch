import numpy 
from numpy.lib.stride_tricks import as_strided

"""

These are the implementations for the convolution functions that are widely used in CNNS. No other library is used other than numpy, purely for optimization. 

Few things to understand: 
    The output of the convolution is governed by the following equation below. 

    output_dim = ((input_dim + 2 * padding - kernel) // stride) + 1

    What are the arguments that are used in these classes? 
    
    kernel: The weight matrix that extracts features from an image (edge, shapes, etc.)
    stride: Defines by how many steps the kernel moves after each element-wise matrix multiplication. 
    padding: The addition of rows and columns around the edges to match the same as the input dimensions. 
    
    The Conv2D and Conv3D simply extend it to further dimensions. Will work on a generalized ConvND as a later exercise
"""

class Conv1D:
    def __init__(self, kernel: int, stride: int, padding: int) -> None: 
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
    
    def forward(self, in_array: numpy.ndarray) -> numpy.ndarray: 
        
        # The forward function defines the convolution operation
        padded_input = numpy.pad(in_array, pad_width=self.padding, mode="constant", constant_values=0)

        out_len = ((len(padded_input) - len(self.kernel)) // self.stride) + 1
        out_array = numpy.ones(out_len)

        for i in range(out_len): 
            idx = i + self.stride
            end = idx + len(self.kernel)
            out_array[i] = numpy.sum(numpy.multiply(padded_input[idx:end], self.kernel))

        return out_array



class Conv2D: 
    def __init__(self, kernel: numpy.ndarray, stride: int, padding: int) -> None: 
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def forward(self, in_matrix: numpy.ndarray) -> numpy.ndarray: 
        i_w, i_h = in_matrix.shape
        k_w, k_h = self.kernel.shape

        padded_input = numpy.pad(in_matrix, ((self.padding, self.padding), (self.padding, self.padding)), mode="constant", constant_values=0)

        o_h = ((i_h + 2*self.padding - k_h) // self.stride) + 1
        o_w = ((i_w + 2*self.padding - k_w) // self.stride) + 1

        out_matrix = numpy.ones((o_w, o_h))

        for i in range(o_w): 
            for j in range(o_h):
                o_i = i * self.stride
                o_j = j * self.stride

                i_end = o_i + len(self.kernel)
                j_end = o_j + len(self.kernel)

                sub_matrix = padded_input[o_i:i_end, o_j:j_end]
                out_matrix[i, j] = numpy.sum(numpy.multiply(sub_matrix, self.kernel))

        return out_matrix


class Conv3D: 
    def __init__(self, kernel: numpy.ndarray, stride: int, padding: int) -> None: 
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def forward(self, in_tensor: numpy.ndarray) -> numpy.ndarray:
        """

        As the number of dimensions increase, it becomes computationally inefficient to use nested loops. Using a triple nested loops is very time consuming as the function runs in O(n^3) time. We can optimize it by using numpy's as_strided library. 

        as_strided() creates a 'view' where the actual array is not initialized nor copied, allowing us to perform operations on the view which would otherwise be difficult or inefficientto implement
        """

        i_d, i_w, i_h = in_tensor.shape
        k_d, k_w, k_h = self.kernel.shape


        padded_input = numpy.pad(in_tensor, ((self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding)), mode="constant", constant_values=0)

        o_d = ((i_d + 2*self.padding - k_d) // self.stride) + 1
        o_h = ((i_h + 2*self.padding - k_h) // self.stride) + 1
        o_w = ((i_w + 2*self.padding - k_w) // self.stride) + 1

        shape = (o_d, o_h, o_w, k_d, k_h, k_w)
        strides = (self.stride * padded_input.strides[0], self.stride * padded_input.strides[1], self.stride * padded_input.strides[2], padded_input[0], padded_input[1], padded_input[2])

        windows = as_strided(padded_input, shape=shape, strides=strides)

        out_tensor = numpy.tensordot(windows, self.kernel, axes=((3, 4, 5), (0, 1, 2)))
        return out_tensor
