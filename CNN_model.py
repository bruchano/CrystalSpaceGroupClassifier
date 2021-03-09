import torch


class ResidualConv2D(torch.nn.Module):
    """
        Residual Convolution Layer
        found to help preventing vanishing gradient problem
    """
    def __init__(self, in_channels, out_channels, kernel_size, dropout, negative_slope, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.conv2 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.dropout = torch.nn.Dropout(dropout)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        norm = torch.nn.LayerNorm(x.shape[3])
        y = self.lrelu(norm(x))
        y = self.conv1(y)
        y = self.dropout(self.lrelu(norm(y)))
        y = self.conv2(y)
        return x + y


class SpaceGroupCNN(torch.nn.Module):
    """
        CNN model
        :param n_cnn: number of residual CNN
        :param channels: number of channels in CNN
        :param out_features: number of output space groups
        :param kernel_size: kernel size for each CNN layer
        :param dropout: dropout percentage of dropout (0 - 1)
        :param negative_slope: negative slope for leaky RELU
        :param *args: dimension for the fully-connected layers (e.g. 2048, 1024, 512, 128)
    """
    def __init__(self, n_cnn, channels, out_features, n_fc=None, kernel_size=3, dropout=0.1, negative_slope=1e-2):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.cnn = torch.nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size)
        self.residualcnn = torch.nn.Sequential(
            *[ResidualConv2D(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                             dropout=dropout, negative_slope=negative_slope) for i in range(n_cnn)]
        )
        self.fc = torch.nn.Sequential()
        if n_fc:
            self.fc.add_module("0", torch.nn.Linear(channels * 46 * 98, n_fc[0]))
            for i in range(len(n_fc) - 1):
                self.fc.add_module(str(i + 1), torch.nn.Linear(n_fc[i], n_fc[i + 1]))
            self.fc.add_module(str(len(n_fc)), torch.nn.Linear(n_fc[-1], out_features))
        else:
            self.fc.add_module("0", torch.nn.Linear(channels * 46 * 98, out_features))
        self.lrelu = torch.nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = x.reshape(1, 1, x.shape[0], x.shape[1])
        size = x.shape
        x = self.lrelu(self.cnn(x))
        norm1 = torch.nn.LayerNorm(x.shape[3])
        x = norm1(x)
        x = self.lrelu(self.residualcnn(x))
        norm2 = torch.nn.LayerNorm(x.shape[3])
        x = norm2(x)
        x = x.view(1, -1)
        x = self.fc(self.dropout(x))
        return torch.softmax(x, 1)

