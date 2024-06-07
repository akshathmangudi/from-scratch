import numpy
from src import conv

# Run these examples if you would like to test the Conv1D operation.
# Similar format can be used for Conv2D and Conv3D if you would like to experiment with that.

in_array = numpy.array([1, 2, 3, 4, 5])
kernel_1 = numpy.random.randn(3, 1)
stride = 1
padding_1 = 0 

kernel_2 = numpy.random.randn(5, 1)
padding_2 = 1

out = conv.Conv1D(kernel=kernel_1, stride=stride, padding=padding_1)
out_1 = out.forward(in_array=in_array)

out2 = conv.Conv1D(kernel=kernel_2, stride=stride, padding=padding_2)
out_2 = out.forward(in_array=out_1)

print(out_2)
