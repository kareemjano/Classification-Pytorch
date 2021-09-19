from .general_nets import conv_pool_relu
from torch import nn

def my_CNN(activ_fn, f_size, f_channels, padding=0, input_shape=(3, 128, 128)):

        input_channels, height, width = (3, 128, 128)
        # input_size = height
        # output_size = int(f_channels * 4 * int(height / 8) * int(width / 8))

        return nn.Sequential(
            conv_pool_relu(input_channels, f_channels, f_size, padding, activ_fn=activ_fn),
            conv_pool_relu(f_channels, f_channels * 2, f_size, padding, activ_fn=activ_fn),
            conv_pool_relu(f_channels * 2, f_channels * 4, f_size, padding, activ_fn=activ_fn),
        )
