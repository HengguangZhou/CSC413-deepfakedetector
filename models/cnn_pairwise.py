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
                                            out_channels=48,
                                            kernel_size=7,
                                            stride=4),
                                  nn.BatchNorm2d(48),
                                  nn.Sigmoid(),
                                  )
        self.conv.apply(init_conv_weights)
        # First dense layer
        self.db1_1 = self.dense_conv(48, 48, 3)
        self.db1_1.apply(init_conv_weights)
        self.db1_2 = self.dense_conv(48*2, 60, 3)
        self.db1_2.apply(init_conv_weights)
        # self.db1_3 = self.dense_conv(48, 60, 3)
        # self.db1_3.apply(init_conv_weights)

        # Second dense layer
        self.db2_1 = self.dense_conv(60, 60, 3)
        self.db2_1.apply(init_conv_weights)
        self.db2_2 = self.dense_conv(60*2, 60, 3)
        self.db2_2.apply(init_conv_weights)
        self.db2_3 = self.dense_conv(60*3, 78, 3)
        self.db2_3.apply(init_conv_weights)
        # self.db2_4 = self.dense_conv(60, 78, 3)
        # self.db2_4.apply(init_conv_weights)

        # Third dense layer
        self.db3_1 = self.dense_conv(78, 78, 3)
        self.db3_1.apply(init_conv_weights)
        self.db3_2 = self.dense_conv(78*2, 78, 3)
        self.db3_2.apply(init_conv_weights)
        self.db3_3 = self.dense_conv(78*3, 78, 3)
        self.db3_3.apply(init_conv_weights)
        self.db3_4 = self.dense_conv(78*4, 99, 3)
        self.db3_4.apply(init_conv_weights)
        # self.db3_5 = self.dense_conv(78, 99, 3)
        # self.db3_5.apply(init_conv_weights)

        # Fourth dense layer
        self.db4_1 = self.dense_conv(99, 99, 3)
        self.db4_1.apply(init_conv_weights)
        self.db4_2 = self.dense_conv(99*2, 99, 3)
        self.db4_2.apply(init_conv_weights)
        self.db4_3 = self.dense_conv(99*3, 171, 3)
        self.db4_3.apply(init_conv_weights)
        # self.db4_4 = self.dense_conv(99, 171, 3)
        # self.db4_4.apply(init_conv_weights

        self.conv2 = nn.Sequential(nn.Conv2d(171, 2, kernel_size=3),
                                   nn.AvgPool2d(23))
        self.conv2.apply(init_conv_weights)

        # 171x9x9
        self.fc1 = nn.Sequential(nn.Linear(171*25*25, 128), nn.Sigmoid())  # Don't know the size for input channels...
        self.fc1.apply(init_fc_weights)
        self.fc2 = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
        self.fc2.apply(init_fc_weights)

    def dense_conv(self, in_channel, out_channel, kernel_size):
        return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, padding=1),
                             nn.BatchNorm2d(out_channel),
                             nn.ReLU())

    # first dense layer forward
    def forward_dense1(self, x):
        first_out_x = self.db1_1(x)
        out = self.db1_2(torch.cat((x, first_out_x), dim=1))
        # out = self.db1_3(torch.cat((x, first_out_x, second_out_x)))
        return out

    def forward_dense2(self, x):
        first_out_x = self.db2_1(x)
        second_out_x = self.db2_2(torch.cat((x, first_out_x), dim=1))
        out = self.db2_3(torch.cat((x, first_out_x, second_out_x), dim=1))
        # out = self.db2_4(torch.cat((x, first_out_x, second_out_x, third_out_x)))

        return out

    def forward_dense3(self, x):
        first_out_x = self.db3_1(x)
        second_out_x = self.db3_2(torch.cat((x, first_out_x), dim=1))
        third_out_x = self.db3_3(torch.cat((x, first_out_x, second_out_x), dim=1))
        out = self.db3_4(torch.cat((x, first_out_x, second_out_x, third_out_x), dim=1))
        # out = self.db3_5(torch.cat((x, first_out_x, second_out_x, third_out_x, fourth_out_x)))

        return out

    def forward_dense4(self, x):
        first_out_x = self.db4_1(x)
        second_out_x = self.db4_2(torch.cat((x, first_out_x), dim=1))
        out = self.db4_3(torch.cat((x, first_out_x, second_out_x), dim=1))
        # out = self.db4_4(torch.cat((x, first_out_x, second_out_x, third_out_x)))

        return out

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.forward_dense1(x1)
        x1 = self.forward_dense2(x1)
        x1 = self.forward_dense3(x1)
        x1 = self.forward_dense4(x1)
        x1_r = x1.view(x1.shape[0], -1)
        x1_r = self.fc1(x1_r)

        # Second image
        x2 = self.conv(x2)
        x2 = self.forward_dense1(x2)
        x2 = self.forward_dense2(x2)
        x2 = self.forward_dense3(x2)
        x2 = self.forward_dense4(x2)
        x2_r = x2.view(x2.shape[0], -1)

        x2 = self.conv2(x2)
        x2_r = self.fc1(x2_r)

        return x1_r, x2_r, self.fc2(x2.view(x2.shape[0], -1))