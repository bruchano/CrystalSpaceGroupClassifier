import torch


'''Residual CNN constructed to prevent vanishing gradient'''
'''Dropout layers are added to generalize the model'''
class ResidualConv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1, negative_slope=1e-2):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.conv2 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        norm = torch.nn.LayerNorm(x.shape[3])
        y = self.dropout1(self.lrelu(norm(x)))
        y = self.conv1(y)
        y = self.dropout2(self.lrelu(norm(y)))
        y = self.conv2(y)
        return x + y


'''CNN model setup'''
'''Optional dimensions of residual CNN and fully-connected network can be added'''
class SpaceGroupCNN(torch.nn.Module):
    def __init__(self, n_cnn, channels, out_features, kernel_size=3, dropout=0.1, negative_slope=1e-2, *args):
        super().__init__()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.cnn = torch.nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size)
        self.residualcnn = torch.nn.Sequential(*[ResidualConv2D(in_channels=channels, out_channels=channels, dropout=dropout, negative_slope=negative_slope) for i in range(n_cnn)])
        self.fc = torch.nn.Sequential()
        if args:
            self.fc.add_module("0", torch.nn.Linear(channels * 1200, args[0]))
            for i in range(len(args) - 1):
                self.fc.add_module(str(i + 1), torch.nn.Linear(args[i], args[i + 1]))
            self.fc.add_module(str(len(args)), torch.nn.Linear(args[-1], out_features))
        else:
            self.fc.add_module("0", torch.nn.Linear(channels * 1200, out_features))
        self.lrelu = torch.nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = x.view(1, 1, -1, 48)
        x = self.lrelu(self.cnn(self.dropout1(x)))
        norm1 = torch.nn.LayerNorm(x.shape[3])
        x = norm1(x)
        x = self.lrelu(self.residualcnn(x))
        norm2 = torch.nn.LayerNorm(x.shape[3])
        x = norm2(x)
        x = x.view(1, -1)
        x = self.fc(self.dropout2(x))
        return torch.softmax(x, 1)

