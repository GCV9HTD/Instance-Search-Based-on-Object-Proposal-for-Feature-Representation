import torch.nn as  nn

from attention_block import ResidualBlock, BN


class AttentionModule_0(nn.Module):
    # input image size is 13*13
    def __init__(self, in_channels, out_channels, size1=(13, 13)):
        super(AttentionModule_0, self).__init__()

        self.first_block = ResidualBlock(in_channels, out_channels)

        self.trunk_branch = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax_block1 = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        )
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax_block2 = nn.Sequential(
            BN(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            BN(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.last_block = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.first_block(x)

        out_trunk = self.trunk_branch(x)

        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax_block1(out_mpool1)
        out_interp1 = self.interpolation1(out_softmax1) + x

        out_softmax2 = self.softmax_block2(out_interp1)
        out = (1 + out_softmax2) * out_trunk

        out_last = self.last_block(out)

        return out_last


class AttentionModule_1(nn.Module):
    # input image size is 26*26
    def __init__(self, in_channels, out_channels, size1=(26, 26), size2=(13, 13)):
        super(AttentionModule_1, self).__init__()

        self.first_block = ResidualBlock(in_channels, out_channels)

        self.trunk_branch = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax_block1 = ResidualBlock(out_channels, out_channels)
        self.skip_connection1 = ResidualBlock(out_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax_block2 = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        )
        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)

        self.softmax_block3 = ResidualBlock(out_channels, out_channels)
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax_block4 = nn.Sequential(
            BN(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            BN(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.last_block = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.first_block(x)

        out_trunk = self.trunk_branch(x)

        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax_block1(out_mpool1)
        out_skip1_connection = self.skip_connection1(out_softmax1)

        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax_block2(out_mpool2)
        out_interp2 = self.interpolation2(out_softmax2) + out_softmax1

        out = out_interp2 + out_skip1_connection
        out_softmax3 = self.softmax_block3(out)
        out_interp1 = self.interpolation1(out_softmax3) + x

        out_softmax4 = self.softmax_block4(out_interp1)

        out = (1 + out_softmax4) * out_trunk

        out_last = self.last_block(out)

        return out_last


class AttentionModule_2(nn.Module):
    # input size is 52*52
    def __init__(self, in_channels, out_channels, size1=(52, 52), size2=(26, 26), size3=(13, 13)):
        super(AttentionModule_2, self).__init__()

        self.first_block = ResidualBlock(in_channels, out_channels)

        self.trunk_branch = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax_block1 = ResidualBlock(out_channels, out_channels)
        self.skip_connection1 = ResidualBlock(out_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax2_blocks = ResidualBlock(out_channels, out_channels)
        self.skip_connection2 = ResidualBlock(out_channels, out_channels)

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax_block3 = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        )
        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)

        self.softmax_block4 = ResidualBlock(out_channels, out_channels)
        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)

        self.softmax_block5 = ResidualBlock(out_channels, out_channels)
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax_block6 = nn.Sequential(
            BN(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            BN(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.last_block = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.first_block(x)

        out_trunk = self.trunk_branch(x)

        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax_block1(out_mpool1)
        out_skip1_connection = self.skip_connection1(out_softmax1)

        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip_connection2(out_softmax2)

        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax_block3(out_mpool3)
        out_interp3 = self.interpolation3(out_softmax3) + out_softmax2

        out = out_interp3 + out_skip2_connection
        out_softmax4 = self.softmax_block4(out)
        out_interp2 = self.interpolation2(out_softmax4) + out_softmax1

        out = out_interp2 + out_skip1_connection
        out_softmax5 = self.softmax_block5(out)
        out_interp1 = self.interpolation1(out_softmax5) + x

        out_softmax6 = self.softmax_block6(out_interp1)
        out = (1 + out_softmax6) * out_trunk

        out_last = self.last_block(out)

        return out_last