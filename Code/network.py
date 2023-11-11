import torch.nn.functional as F
import torch.nn as nn
import torch


def initial_layer(in_dim, out_dim_pre, out_dim):
    return nn.Sequential(nn.Conv3d(in_dim, out_dim_pre, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm3d(out_dim_pre), nn.ReLU(inplace=True),
                         nn.Conv3d(out_dim_pre, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm3d(out_dim), nn.ReLU(inplace=True))


def conv_block_layer_en(in_dim, out_dim):
    return nn.Sequential(nn.Conv3d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm3d(in_dim), nn.ReLU(inplace=True),
                         nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm3d(out_dim), nn.ReLU(inplace=True))


def max_pool_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_trans_block_3d(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        nn.ReLU(inplace=True))


class Unet3D(nn.Module):
    def __init__(self, in_dim=4, out_dim=4, num_filters=64):
        super(Unet3D, self).__init__()
        self.in_dim = in_dim
        self.num_filters = num_filters
        self.out_dim = out_dim

        self.conv1 = initial_layer(self.in_dim, 32, self.num_filters)
        self.pool1 = max_pool_3d()

        self.conv2 = conv_block_layer_en(self.num_filters, self.num_filters * 2)
        self.pool2 = max_pool_3d()

        self.conv3 = conv_block_layer_en(self.num_filters * 2, self.num_filters * 4)
        self.pool3 = max_pool_3d()

        self.bridge = conv_block_layer_en(self.num_filters * 4, self.num_filters * 8)

        self.upconv2 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8)  # 512
        self.dconv3 = conv_block_layer_en(self.num_filters * 12, self.num_filters * 4)  # 512 + 256 | 256

        self.upconv3 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4)  # 256
        self.dconv2 = conv_block_layer_en(self.num_filters * 6, self.num_filters * 2)  # 256 + 128 | 128

        self.upconv4 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2)  # 128
        self.dconv1 = conv_block_layer_en(self.num_filters * 3, self.num_filters * 1)  # 128 + 64 | 64

        self.final_conv = nn.Sequential(nn.Conv3d(self.num_filters, self.out_dim, kernel_size=3, padding=1))

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        bridge = self.bridge(pool3)

        trans_2 = self.upconv2(bridge)
        concat_2 = torch.cat([trans_2, conv3], dim=1)
        dconv3 = self.dconv3(concat_2)

        trans_3 = self.upconv3(dconv3)
        concat_3 = torch.cat([trans_3, conv2], dim=1)
        dconv2 = self.dconv2(concat_3)

        trans_4 = self.upconv4(dconv2)
        concat_2 = torch.cat([trans_4, conv1], dim=1)
        dconv1 = self.dconv1(concat_2)

        x = self.final_conv(dconv1)
        x = F.softmax(x, dim=1)
        return x