import torch
import torch.nn as nn
import torch.functional as F


# adapted from https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
def init_conv_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight, std=0.1)
        torch.nn.init.normal_(m.bias, mean=0.5, std=0.1)

def init_fc_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=2)
        torch.nn.init.normal_(m.bias, mean=0.5, std=0.1)


class CnnPairwise(nn.Module):
    def __init__(self, input_channel):
        super(CnnPairwise, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=input_channel,
                                            out_channels=96,
                                            kernel_size=7,
                                            stride=4),
                                  nn.BatchNorm2d(96),
                                  nn.ReLU(),
                                  )
        self.conv.apply(init_conv_weights)
        self.db1 = nn.Sequential(nn.Conv2d(96, 96, 3),
                                 nn.Sigmoid(),
                                 nn.Conv2d(96, 128, 3),
                                 nn.Sigmoid())
        self.db2 = nn.Sequential(nn.Conv2d(128, 128, 3),
                                 nn.Sigmoid(),
                                 nn.Conv2d(128, 256, 3),
                                 nn.Sigmoid())
        self.db3 = nn.Sequential(nn.Conv2d(256, 256, 3),
                                 nn.Sigmoid(),
                                 nn.Conv2d(256, 256, 3),
                                 nn.Sigmoid())
        self.conv2 = nn.Sequential(nn.Conv2d(256, 2, kernel_size=3),
                                   nn.AvgPool2d(3))

        self.fc1 = nn.Sequential(nn.Linear(9216, 128), nn.Softmax())  # Don't know the size for input channels...
        self.fc1.apply(init_fc_weights)
        self.fc2 = nn.Sequential(nn.Linear(2, 2), nn.Softmax())
        self.fc2.apply(init_fc_weights)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.db1(x1)
        x1 = self.db2(x1)
        x1 = self.db3(x1)
        x2 = self.conv(x2)
        x2 = self.db1(x2)
        x2 = self.db2(x2)
        x2 = self.db3(x2)
        x1 = self.fc1(x1.view(x1.shape[0], -1))
        x2 = self.fc1(x2.view(x2.shape[0], -1))
        return self.fc2(torch.abs(x1 - x2))