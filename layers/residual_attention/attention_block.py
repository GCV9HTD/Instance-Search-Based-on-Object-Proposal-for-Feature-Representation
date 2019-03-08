import torch
import torch.nn as nn

def BN(inplanes, group=1):

    if group == 1:
        return nn.BatchNorm2d(inplanes)
    else:
        return nn.GroupNorm(group, inplanes)


class ResidualBlock(nn.Module):

    def __init__(self, input_channels, output_channels, stride=1, relu=True):
        super(ResidualBlock, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(input_channels)
        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels / 4, 1, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(output_channels / 4)
        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(output_channels / 4, output_channels / 4, 3, stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(output_channels / 4)
        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv3 = nn.Conv2d(output_channels / 4, output_channels, 1, 1, bias=False)

        self.conv4 = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)

    def forward(self, x):

        residual = x

        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(out1)

        out += residual

        return out
