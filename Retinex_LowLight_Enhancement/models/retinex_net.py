
import torch
import torch.nn as nn

from models.decom_net import DecomNet
from models.relight_net import RelightNet


class RetinexNet(nn.Module):

    def __init__(self):
        super(RetinexNet, self).__init__()

        self.decom_net = DecomNet()
        self.relight_net = RelightNet()

    def forward(self, input_low):

        R_low, L_low = self.decom_net(input_low)

        I_delta = self.relight_net(L_low, R_low)

        I_delta_3 = I_delta.repeat(1, 3, 1, 1)

        enhanced = R_low * I_delta_3

        return {
            "R_low": R_low,
            "L_low": L_low,
            "I_delta": I_delta,
            "enhanced": enhanced
        }
