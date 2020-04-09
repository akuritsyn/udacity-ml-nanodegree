import torch
import torch.nn as nn
from torch.nn import functional as F


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        # if not (target.size() == input.size()):
        #     raise ValueError("Target size ({}) must be the same as input size ({})"
        #                      .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1.0

    def forward(self, input, target):
        intersection = 2.0 * ((target * input).sum()) + self.smooth
        union = target.sum() + input.sum() + self.smooth
        return 1 - (intersection / union)


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.diceloss = DiceLoss()
        self.bceloss = nn.BCELoss()

    def forward(self, input, target):
        input = torch.sigmoid(input)
        return self.bceloss(input, target) + self.diceloss(input, target)
