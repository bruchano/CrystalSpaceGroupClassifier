import torch


class FullyConnected(torch.nn.Module):
    """
        Custom changes on fully-connected
        dropout layer was found to help generalizing the model
    """
    def __init__(self, in_features, out_features, dropout, negative_slope):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(in_features)
        self.lrelu = torch.nn.LeakyReLU(negative_slope)
        self.fc = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.norm(self.dropout(self.lrelu(x)))
        return self.fc(x)



class SpaceGroupNN(torch.nn.Module):
    """
        NN Model
        :param out_features: number of output space groups
        :param dropout: dropout percentage of dropout (0 - 1)
        :param negative_slope: negative slope for leaky RELU
        :param *args: dimension for the fully-connected layers (e.g. 2048, 1024, 512, 128)
    """
    def __init__(self, out_features, dropout=0.1, negative_slope=1e-2, *args):
        super().__init__()
        self.fc = torch.nn.Sequential()
        if args:
            self.fc.add_module("0", FullyConnected(1200, args[0], dropout, negative_slope))
            for i in range(len(args) - 1):
                self.fc.add_module(str(i + 1), FullyConnected(args[i], args[i + 1], dropout, negative_slope))
            self.fc.add_module(str(len(args)), FullyConnected(args[-1], out_features, dropout, negative_slope))
        else:
            self.fc.add_module("0", FullyConnected(1200, out_features, dropout, negative_slope))

    def forward(self, x):
        x = x.view(1, -1)
        x = self.fc(x)
        return torch.softmax(x, 1)

