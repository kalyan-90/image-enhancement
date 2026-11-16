
import torch
import torch.nn as nn


class DecomNet(nn.Module):
    def __init__(self, layer_num=5, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()

        self.shallow_feature_extraction = nn.Conv2d(
            4,
            channel,
            kernel_size * 3,
            padding=(kernel_size * 3) // 2
        )

        self.deep_feature_extraction = nn.Sequential()

        for i in range(layer_num):
            self.deep_feature_extraction.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    channel,
                    channel,
                    kernel_size,
                    padding=kernel_size // 2
                )
            )

            self.deep_feature_extraction.add_module(
                f"relu_{i}",
                nn.ReLU(inplace=True)
            )

        self.recon = nn.Conv2d(
            channel,
            4,
            kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, input_im):

        input_max = torch.max(
            input_im,
            dim=1,
            keepdim=True
        )[0]

        input_img = torch.cat(
            (input_max, input_im),
            dim=1
        )

        feats0 = self.shallow_feature_extraction(input_img)

        featss = self.deep_feature_extraction(feats0)

        outs = self.recon(featss)

        R = torch.sigmoid(outs[:, 0:3, :, :])
        L = torch.sigmoid(outs[:, 3:4, :, :])

        return R, L
