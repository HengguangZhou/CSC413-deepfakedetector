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

class siamese(nn.Module):
    def __init__(self, input_channel):
        super(siamese, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=input_channel,
                                            out_channels=64,
                                            kernel_size=10),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2),
                                  nn.Conv2d(in_channels=64,
                                            out_channels=128,
                                            kernel_size=7),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2),
                                  nn.Conv2d(in_channels=128,
                                            out_channels=128,
                                            kernel_size=4),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2),
                                  nn.Conv2d(in_channels=128,
                                            out_channels=256,
                                            kernel_size=4),
                                  nn.ReLU()
                                  )
        self.conv.apply(init_conv_weights)
        self.fc1 = nn.Linear(73984, 4096)
        self.fc1.apply(init_fc_weights)
        self.fc2 = nn.Linear(4096, 1)
        self.fc2.apply(init_fc_weights)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.fc1(x1.view(x1.shape[0], -1))
        x1 = torch.sigmoid(x1)
        x2 = self.conv(x2)
        x2 = self.fc1(x2.view(x2.shape[0], -1))
        x2 = torch.sigmoid(x2)
        return self.fc2(torch.abs(x1 - x2))
