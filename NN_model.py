import torch


'''Custom changes on the fully-connected network'''
class FullyConnected(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, negative_slope):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(in_features)
        self.lrelu = torch.nn.LeakyReLU(negative_slope)
        self.fc = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.norm(self.dropout(self.lrelu(x)))
        return self.fc(x)


'''NN model setup'''
''''''
class SpaceGroupNN(torch.nn.Module):
    def __init__(self, out_features, dropout, negative_slope, *args):
        super().__init__()
        self.fc = torch.nn.Sequential()
        if args:
            self.fc.add_module("0", FullyConnected(1200, args[0], dropout, negative_slope))
            for i in range(len(args) - 1):
                self.fc.add_module(str(i + 1), FullyConnected(args[i], args[i + 1], dropout, negative_slope))
            self.fc.add_module(str(len(args)), FullyConnected(args[-1], out_features, dropout, negative_slope))
        else:
            self.fc.add_module("0", torch.nn.Linear(1200, out_features, dropout))

    def forward(self, x):
        x = x.view(1, -1)
        x = self.fc(x)
        return torch.softmax(x, 1)

