import torch
import torch.nn as nn


class LeNet(nn.Module):

    def __init__(self, n_classes):
        super(LeNet, self).__init__()

        self.lenet_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6,
                      kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120,
                      kernel_size=5, stride=1),
            nn.ReLU(),
        )

        self.output_layers = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.lenet_layers(x)
        x = torch.flatten(x, 1)
        res = self.output_layers(x)
        return res


class AlexNet(nn.Module):
    def __init__(self, n_classes):
        super(AlexNet, self).__init__()

        self.alexnet_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96,
                      kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=96, out_channels=256,
                      kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=256, out_channels=384,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384,
                      kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.output_layers = nn.Sequential(
            nn.Linear(256*5*5,  4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.alexnet_layers(x)
        x = torch.flatten(x, 1)
        res = self.output_layers(x)
        return res


class VGGNet(nn.Module):
    def __init__(self, n_classes):
        super(VGGNet, self).__init__()

        def vgg_block(num_convs, in_channels, out_channels):
            layers = []
            for _ in range(num_convs):
                layers.append(nn.Conv2d(in_channels, out_channels,
                                        kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        def vgg(architecture):
            conv_blocks = []
            in_channels = 1

            for (num_convs, out_channels) in architecture:
                conv_blocks.append(
                    vgg_block(num_convs, in_channels, out_channels))
                in_channels = out_channels

            return nn.Sequential(*conv_blocks)

        architecture = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        self.vgg_layers = vgg(architecture)

        self.output_layers = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.vgg_layers(x)
        x = torch.flatten(x, 1)
        res = self.output_layers(x)
        return res


class ResidualLayer(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, num_channels,
                      kernel_size=3, padding=1, stride=strides),
            nn.BatchNorm2d(num_channels), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels)
        )
        self.conv1x1 = nn.Conv2d(
            input_channels, num_channels, kernel_size=1, stride=strides)
        self.use_1x1conv = use_1x1conv

    def forward(self, x):
        y = self.layers(x)
        if self.use_1x1conv:
            x = self.conv1x1(x)
        y += x
        return nn.ReLU()(y)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(ResidualLayer(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(ResidualLayer(num_channels, num_channels))
    return blk


class ResNet(nn.Module):
    def __init__(self, n_classes):
        super(ResNet, self).__init__()
        self.header = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.resnet = nn.Sequential(
            *self.header,
            *resnet_block(64, 64, 2, first_block=True),
            *resnet_block(64, 128, 2),
            *resnet_block(128, 256, 2),
            *resnet_block(256, 512, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.output_layer = nn.Sequential(
            nn.Linear(512, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        res = self.output_layer(x)
        return res
