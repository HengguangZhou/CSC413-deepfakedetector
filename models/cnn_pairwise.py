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
                                  nn.Sigmoid(),
                                  )
        self.conv.apply(init_conv_weights)
        self.db1 = self.dense_block(96, 128, 3)
        self.db1.apply(init_conv_weights)
        self.db2 = self.dense_block(128, 256, 3)
        self.db2.apply(init_conv_weights)
        self.db3 = self.dense_block(256, 256, 3)
        self.db3.apply(init_conv_weights)

        self.conv2 = nn.Sequential(nn.Conv2d(256, 2, kernel_size=3),
                        nn.AvgPool2d(11))
        self.conv2.apply(init_conv_weights)

        self.fc1 = nn.Sequential(nn.Linear(256*13*13, 128), nn.Sigmoid())  # Don't know the size for input channels...
        self.fc1.apply(init_fc_weights)
        self.fc2 = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
        self.fc2.apply(init_fc_weights)

    def dense_block(self, in_channel, out_channel, kernel_size):
        return nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size),
                             nn.BatchNorm2d(in_channel),
                             nn.Sigmoid(),
                             nn.Conv2d(in_channel, out_channel, kernel_size),
                             nn.BatchNorm2d(out_channel),
                             nn.Sigmoid())

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.db1(x1)
        x1 = self.db2(x1)
        x1 = self.db3(x1)
        x1_r = x1.view(x1.shape[0], -1)
        x1_r = self.fc1(x1_r)

        # Second image
        x2 = self.conv(x2)
        x2 = self.db1(x2)
        x2 = self.db2(x2)
        x2 = self.db3(x2)
        x2_r = x2.view(x2.shape[0], -1)

        x2 = self.conv2(x2)
        x2_r = self.fc1(x2_r)

        return x1_r, x2_r, self.fc2(x2.view(x2.shape[0], -1))
