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
        # self.db1 = self.dense_block1(48, 60, 3)
        # self.db1.apply(init_conv_weights)
        #
        # self.db2 = self.dense_block(4, 60, 78, 3)
        # self.db2.apply(init_conv_weights)
        #
        # self.db3 = self.dense_block(5, 78, 99, 3)
        # self.db3.apply(init_conv_weights)
        #
        # self.db4 = self.dense_block(4, 99, 171, 3)
        # self.db4.apply(init_conv_weights)

        self.conv2 = nn.Sequential(nn.Conv2d(171, 2, kernel_size=3),
                                   nn.AvgPool2d(23))
        self.conv2.apply(init_conv_weights)

        # 171x9x9
        self.fc1 = nn.Sequential(nn.Linear(171*25*25, 128), nn.Sigmoid())  # Don't know the size for input channels...
        self.fc1.apply(init_fc_weights)
        self.fc2 = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
        self.fc2.apply(init_fc_weights)

    def conv_for_dense_block(self, in_channel, out_channel, kernel_size):
        return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, padding=1),
                             nn.BatchNorm2d(out_channel),
                             nn.Sigmoid())

    # 3 layer dense block
    def forward_dense1(self, x):
        conv = self.conv_for_dense_block(48, 48, 3).to("cuda")
        first_out_x = conv(x)
        # conv = self.conv_for_dense_block(48, 48, 3).to("cuda")
        # second_out_x = conv(torch.cat((x, first_out_x)))
        conv = self.conv_for_dense_block(48, 60, 3).to("cuda")
        second_out_x = conv(torch.cat((x, first_out_x)))

        return second_out_x

    def forward_dense2(self, x):
        conv = self.conv_for_dense_block(60, 60, 3).to("cuda")
        first_out_x = conv(x)
        # conv = self.conv_for_dense_block(60, 60, 3).to("cuda")
        # second_out_x = conv(torch.cat((x, first_out_x)))
        # conv = self.conv_for_dense_block(60, 60, 3).to("cuda")
        # third_out_x = conv(torch.cat((x, first_out_x, second_out_x)))
        conv = self.conv_for_dense_block(60, 78, 3).to("cuda")
        second_out_x = conv(torch.cat((x, first_out_x)))

        return second_out_x

    def forward_dense3(self, x):
        conv = self.conv_for_dense_block(78, 78, 3).to("cuda")
        first_out_x = conv(x)
        # conv = self.conv_for_dense_block(78, 78, 3).to("cuda")
        # second_out_x = conv(torch.cat((x, first_out_x)))
        # conv = self.conv_for_dense_block(78, 78, 3).to("cuda")
        # third_out_x = conv(torch.cat((x, first_out_x, second_out_x)))
        # conv = self.conv_for_dense_block(78, 78, 3).to("cuda")
        # fourth_out_x = conv(torch.cat((x, first_out_x, second_out_x, third_out_x)))
        conv = self.conv_for_dense_block(78, 99, 3).to("cuda")
        second_out_x = conv(torch.cat((x, first_out_x)))

        return second_out_x

    def forward_dense4(self, x):
        conv = self.conv_for_dense_block(99, 99, 3).to("cuda")
        first_out_x = conv(x)
        # conv = self.conv_for_dense_block(99, 99, 3).to("cuda")
        # second_out_x = conv(torch.cat((x, first_out_x)))
        # conv = self.conv_for_dense_block(99, 99, 3).to("cuda")
        # third_out_x = conv(torch.cat((x, first_out_x, second_out_x)))
        conv = self.conv_for_dense_block(99, 171, 3).to("cuda")
        second_out_x = conv(torch.cat((x, first_out_x)))

        return second_out_x

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