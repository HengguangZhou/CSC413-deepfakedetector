import torch
import torch.nn as nn
import torch.nn.functional as F


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
    # def __init__(self, input_channel):
    #     super(CnnPairwise, self).__init__()
    #     self.conv = nn.Sequential(nn.Conv2d(in_channels=input_channel,
    #                                         out_channels=48,
    #                                         kernel_size=7,
    #                                         stride=4),
    #                               nn.BatchNorm2d(48),
    #                               nn.SiLU(),
    #                               )
    #     self.conv.apply(init_conv_weights)
    #     # First dense layer
    #     self.db1_1 = self.dense_conv(48, 48, 3)
    #     self.db1_1.apply(init_conv_weights)
    #     self.db1_2 = self.dense_conv(48*2, 48, 3)
    #     self.db1_2.apply(init_conv_weights)
    #     self.db1_3 = self.dense_conv(48*3, 60, 3)
    #     self.db1_3.apply(init_conv_weights)
    #
    #     # Second dense layer
    #     self.db2_1 = self.dense_conv(60, 60, 3)
    #     self.db2_1.apply(init_conv_weights)
    #     self.db2_2 = self.dense_conv(60*2, 60, 3)
    #     self.db2_2.apply(init_conv_weights)
    #     self.db2_3 = self.dense_conv(60*3, 60, 3)
    #     self.db2_3.apply(init_conv_weights)
    #     self.db2_4 = self.dense_conv(60*4, 78, 3)
    #     self.db2_4.apply(init_conv_weights)
    #
    #     # Third dense layer
    #     self.db3_1 = self.dense_conv(78, 78, 3)
    #     self.db3_1.apply(init_conv_weights)
    #     self.db3_2 = self.dense_conv(78*2, 78, 3)
    #     self.db3_2.apply(init_conv_weights)
    #     self.db3_3 = self.dense_conv(78*3, 78, 3)
    #     self.db3_3.apply(init_conv_weights)
    #     self.db3_4 = self.dense_conv(78*4, 78, 3)
    #     self.db3_4.apply(init_conv_weights)
    #     self.db3_5 = self.dense_conv(78*5, 99, 3)
    #     self.db3_5.apply(init_conv_weights)
    #
    #     # Fourth dense layer
    #     self.db4_1 = self.dense_conv(99, 99, 3)
    #     self.db4_1.apply(init_conv_weights)
    #     self.db4_2 = self.dense_conv(99*2, 99, 3)
    #     self.db4_2.apply(init_conv_weights)
    #     self.db4_3 = self.dense_conv(99*3, 99, 3)
    #     self.db4_3.apply(init_conv_weights)
    #     self.db4_4 = self.dense_conv(99*4, 171, 3)
    #     self.db4_4.apply(init_conv_weights)
    #
    #     self.conv2 = nn.Sequential(nn.Conv2d(171, 2, kernel_size=3),
    #                                nn.BatchNorm2d(2),
    #                                nn.AvgPool2d(23),
    #                                nn.SiLU())
    #     self.conv2.apply(init_conv_weights)
    #
    #     # 171x9x9
    #     self.fc1 = nn.Sequential(nn.Linear(171*25*25, 128), nn.SiLU())  # Don't know the size for input channels...
    #     self.fc1.apply(init_fc_weights)
    #     self.fc2 = nn.Sequential(nn.Linear(2, 1), nn.Softmax())
    #     self.fc2.apply(init_fc_weights)
    #
    # def dense_conv(self, in_channel, out_channel, kernel_size):
    #     return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, padding=1),
    #                          nn.BatchNorm2d(out_channel),
    #                          nn.ReLU())
    #
    # # first dense layer forward
    # def forward_dense1(self, x):
    #     first_out_x = self.db1_1(x)
    #     second_out_x = self.db1_2(torch.cat((x, first_out_x), dim=1))
    #     out = self.db1_3(torch.cat((x, first_out_x, second_out_x), dim=1))
    #     return out
    #
    # def forward_dense2(self, x):
    #     first_out_x = self.db2_1(x)
    #     second_out_x = self.db2_2(torch.cat((x, first_out_x), dim=1))
    #     third_out_x = self.db2_3(torch.cat((x, first_out_x, second_out_x), dim=1))
    #     out = self.db2_4(torch.cat((x, first_out_x, second_out_x, third_out_x), dim=1))
    #
    #     return out
    #
    # def forward_dense3(self, x):
    #     first_out_x = self.db3_1(x)
    #     second_out_x = self.db3_2(torch.cat((x, first_out_x), dim=1))
    #     third_out_x = self.db3_3(torch.cat((x, first_out_x, second_out_x), dim=1))
    #     fourth_out_x = self.db3_4(torch.cat((x, first_out_x, second_out_x, third_out_x), dim=1))
    #     out = self.db3_5(torch.cat((x, first_out_x, second_out_x, third_out_x, fourth_out_x), dim=1))
    #
    #     return out
    #
    # def forward_dense4(self, x):
    #     first_out_x = self.db4_1(x)
    #     second_out_x = self.db4_2(torch.cat((x, first_out_x), dim=1))
    #     third_out_x = self.db4_3(torch.cat((x, first_out_x, second_out_x), dim=1))
    #     out = self.db4_4(torch.cat((x, first_out_x, second_out_x, third_out_x), dim=1))
    #
    #     return out
    #
    # def forward(self, x1, x2):
    #     x1 = self.conv(x1)
    #     x1 = self.forward_dense1(x1)
    #     x1 = self.forward_dense2(x1)
    #     x1 = self.forward_dense3(x1)
    #     x1 = self.forward_dense4(x1)
    #     x1_r = x1.view(x1.shape[0], -1)
    #     x1_r = self.fc1(x1_r)
    #
    #     # Second image
    #     x2 = self.conv(x2)
    #     x2 = self.forward_dense1(x2)
    #     x2 = self.forward_dense2(x2)
    #     x2 = self.forward_dense3(x2)
    #     x2 = self.forward_dense4(x2)
    #     x2_r = x2.view(x2.shape[0], -1)
    #
    #     x2 = self.conv2(x2)
    #     x2_r = self.fc1(x2_r)
    #
    #     return x1_r, x2_r, self.fc2(x2.view(x2.shape[0], -1))
    def __init__(self, input_channel):
        super(CnnPairwise, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channel,
                              out_channels=96,
                              kernel_size=7,
                              stride=4)
        self.conv.apply(init_conv_weights)
        # First dense layer
        self.db1_1 = ResidualBlock(96, 96, 3, False)
        self.db1_1.apply(init_conv_weights)
        self.db1_2 = ResidualBlock(96, 128, 3, True)
        self.db1_2.apply(init_conv_weights)

        # Second dense layer
        self.db2_1 = ResidualBlock(128, 128, 3, False)
        self.db2_1.apply(init_conv_weights)
        self.db2_2 = ResidualBlock(128, 256, 3, True)
        self.db2_2.apply(init_conv_weights)

        # Third dense layer
        self.db3_1 = ResidualBlock(256, 256, 3, False)
        self.db3_1.apply(init_conv_weights)
        self.db3_2 = ResidualBlock(256, 256, 3, True)
        self.db3_2.apply(init_conv_weights)

        self.conv2 = nn.Sequential(nn.BatchNorm2d(256),
                                   nn.SiLU(),
                                   nn.Conv2d(256, 2, kernel_size=3, padding=1),
                                   nn.AvgPool2d(4))
        self.conv2.apply(init_conv_weights)

        # 256x4x4 after down-sampling
        self.fc1 = nn.Sequential(nn.Linear(256*4*4, 128), nn.Softmax())  # Don't know the size for input channels...
        self.fc1.apply(init_fc_weights)
        self.fc2 = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
        self.fc2.apply(init_fc_weights)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.8)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.db1_1(x1)
        x1 = self.db1_2(x1)
        x1 = self.db2_1(x1)
        x1 = self.db2_2(x1)
        x1 = self.db3_1(x1)
        x1 = self.db3_2(x1)
        x1_r = x1.view(x1.shape[0], -1)
        x1_r = self.dropout1(self.fc1(x1_r))

        # Second image
        x2 = self.conv(x2)
        x2 = self.db1_1(x2)
        x2 = self.db1_2(x2)
        x2 = self.db2_1(x2)
        x2 = self.db2_2(x2)
        x2 = self.db3_1(x2)
        x2 = self.db3_2(x2)
        x2_r = x2.view(x2.shape[0], -1)

        x2 = self.conv2(x2)
        x2_r = self.dropout1(self.fc1(x2_r))

        pred = self.dropout2(self.fc2(x2.view(x2.shape[0], -1)))

        return x1_r, x2_r, pred


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, down_sample):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.BatchNorm2d(in_channel),
                                   nn.SiLU(),
                                   nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=1))
        self.conv2 = nn.Sequential(nn.BatchNorm2d(in_channel),
                                   nn.SiLU(),
                                   nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=1))
        self.down_sample = nn.AvgPool2d(kernel_size=1, stride=2) if down_sample else nn.Identity()

    def shortcut(self, x, z):
        if x.shape[1] != z.shape[1]:  # Different number of channels
            diff = z.shape[1] - x.shape[1]
            left_pad = diff // 2
            right_pad = diff - left_pad
            identity = F.pad(x, (0, 0, 0, 0, left_pad, right_pad))
            return z + identity
        else:
            return z + x

    def forward(self, x):
        x = self.down_sample(x)
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.shortcut(x, z)

        return z
