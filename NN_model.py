import torch


class FullyConnected(torch.nn.Module):
    """
        Custom changes on fully-connected
        dropout layer was found to help generalizing the model
    """
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(in_features)
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.norm(self.dropout(self.gelu(x)))
        return self.fc(x)



class SpaceGroupNN(torch.nn.Module):
    """
        NN Model
        :param out_features: number of output space groups
        :param dropout: dropout percentage of dropout (0 - 1)
        :param negative_slope: negative slope for leaky RELU
        :param *args: dimension for the fully-connected layers (e.g. 2048, 1024, 512, 128)
    """
    def __init__(self, out_features, n_fc, dropout=0.05):
        super().__init__()
        self.fc = torch.nn.Sequential()
        self.fc.add_module("0", FullyConnected(48 * 100, n_fc[0], dropout))
        for i in range(len(n_fc) - 1):
            self.fc.add_module(str(i + 1), FullyConnected(n_fc[i], n_fc[i + 1], dropout))
        self.fc.add_module(str(len(n_fc)), FullyConnected(n_fc[-1], out_features, dropout))

    def forward(self, x):
        x = x.view(1, -1)
        x = self.fc(x)
        return torch.softmax(x, 1)

