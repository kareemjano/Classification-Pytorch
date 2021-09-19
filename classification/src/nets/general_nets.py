from torch import nn

def conv_pool_relu(in_channels, out_channels, kernel_size, padding, activ_fn=nn.PReLU):

        model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.MaxPool2d(2, 2),
            activ_fn(),
            nn.BatchNorm2d(out_channels),
        )

        model.apply(weights_init)
        return model

def FCN_layer(input_size, output_size, dropout=0):

    model = nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.PReLU(),
        nn.BatchNorm1d(output_size),
        nn.Dropout(p=dropout),
    )

    model.apply(weights_init)

    return model


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, gain=2.0)