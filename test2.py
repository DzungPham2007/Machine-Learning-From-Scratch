import numpy as np

# b:2, c:3, f:4, h:5, w:5, k:3, t:3
a = np.array([
    [[[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]],

    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]],

    [[1, 1, 1],
     [2, 2, 2],
     [3, 3, 3]],
    
    [[2, 2, 2],
     [2, 2, 2],
     [2, 2, 2]]],

    [[[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]],

    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]],

    [[1, 1, 1],
     [2, 2, 2],
     [3, 3, 3]],
    
    [[2, 2, 2],
     [2, 2, 2],
     [2, 2, 2]]]
])

b = np.array([
    [[[1, 1],
     [1, 1]],

    [[1, 0],
     [0, 1]],

    [[0, 0],
     [1, 1]]],

    [[[1, 1],
     [1, 1]],

    [[1, 0],
     [0, 1]],

    [[0, 0],
     [1, 1]]]
])

print(a.shape, b.shape)
inputs = a
filters = b
stride = 1
# filters.shape = (num_filter, channel, kernel_height, kernel_width)
num_filter = filters.shape[-3]
kernel_height = filters.shape[-2]
kernel_width = filters.shape[-1]

# inputs.shape = (batch_size, channel, height, width)
batch, channel, height, width = inputs.shape

# inputs.strides = (batch_strides, row_stride, columns_strides)
# output is the num of bit need to move to next row/column/depth
batch_stride, channel_stride, row_stride, column_stride = inputs.strides

output_height = int((height - kernel_height) / stride + 1)
output_width = int((width - kernel_width) / stride + 1)

# we divided each inputs into new matrix (output_height x output_width) each elements size kernel
new_shape = (batch, channel, output_height, output_width, kernel_height, kernel_width)

# bit need to move between row/column/depth of new matrix
new_stride = (batch_stride, channel_stride, row_stride * stride, column_stride * stride, row_stride, column_stride)

new_input = np.lib.stride_tricks.as_strided(inputs, new_shape, new_stride)

result = np.einsum("bchwkt,bfkt->bcfhw", new_input, filters)

print(result.shape)
print(result)