import torch
from torch import nn


class Focal_Loss(nn.Module):
    def __init__(self, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        loss = -(1 - inputs) ** self.gamma * targets * torch.log(inputs) - inputs ** self.gamma * (1 - targets) * torch.log(1 - inputs)
        loss = loss.mean()
        return loss

