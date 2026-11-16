
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelightNet(nn.Module):

    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()

        padding = kernel_size // 2

        self.conv0 = nn.Conv2d(
            4, channel, kernel_size, padding=padding
        )

        self.conv1 = nn.Conv2d(
            channel, channel, kernel_size,
            stride=2, padding=padding
        )

        self.conv2 = nn.Conv2d(
            channel, channel, kernel_size,
            stride=2, padding=padding
        )

        self.conv3 = nn.Conv2d(
            channel, channel, kernel_size,
            stride=2, padding=padding
        )

        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.Conv2d(
            channel, channel, kernel_size, padding=padding
        )

        self.deconv2 = nn.Conv2d(
            channel, channel, kernel_size, padding=padding
        )

        self.deconv3 = nn.Conv2d(
            channel, channel, kernel_size, padding=padding
        )

        self.feature_fusion = nn.Conv2d(
            channel * 3, channel, 1
        )

        self.output = nn.Conv2d(
            channel, 1, 3, padding=1
        )

    def forward(self, input_L, input_R):

        input_im = torch.cat((input_R, input_L), dim=1)

        conv0 = self.conv0(input_im)

        conv1 = self.relu(self.conv1(conv0))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(conv2))

        up1 = F.interpolate(
            conv3, size=conv2.shape[2:], mode="nearest"
        )
        deconv1 = self.relu(self.deconv1(up1)) + conv2

        up2 = F.interpolate(
            deconv1, size=conv1.shape[2:], mode="nearest"
        )
        deconv2 = self.relu(self.deconv2(up2)) + conv1

        up3 = F.interpolate(
            deconv2, size=conv0.shape[2:], mode="nearest"
        )
        deconv3 = self.relu(self.deconv3(up3)) + conv0

        deconv1_resize = F.interpolate(
            deconv1, size=deconv3.shape[2:], mode="nearest"
        )

        deconv2_resize = F.interpolate(
            deconv2, size=deconv3.shape[2:], mode="nearest"
        )

        feature_gather = torch.cat(
            (deconv1_resize, deconv2_resize, deconv3),
            dim=1
        )

        feature_fusion = self.feature_fusion(feature_gather)

        output = self.output(feature_fusion)

        return torch.sigmoid(output)
