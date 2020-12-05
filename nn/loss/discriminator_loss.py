import torch
import torch.nn as nn


class DiscriminatorLoss(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, sr_labels, hr_labels):
        hr_loss = self.criterion(hr_labels, torch.ones_like(hr_labels))
        sr_loss = self.criterion(sr_labels, torch.zeros_like(sr_labels))
        return hr_loss + sr_loss
