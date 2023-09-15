import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self,
                 gamma=2,
                 alpha=0.25,
                 reduction='mean',
                 ignore_index=-100,
                 epsilon=1e-16):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_lb = ignore_index
        self.epsilon = epsilon

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        with torch.no_grad():
            label = label.clone().detach().long()
            ignore = label == self.ignore_lb
            ignore_one_hot = ignore.unsqueeze(1).expand_as(logits)
            n_valid = (ignore == 0).sum(dim=(1, 2))
            label[ignore] = 0
            lb_one_hot = torch.zeros_like(logits).scatter_(
                1, label.unsqueeze(1), 1).detach()
            lb_one_hot[ignore_one_hot] = 0

        # Adapted from
        # https://kornia.readthedocs.io/en/v0.1.2/_modules/
        # torchgeometry/losses/focal.html
        probs = torch.softmax(logits, dim=1) + self.epsilon

        weight = torch.pow(1. - probs, self.gamma)
        focal = -self.alpha * weight * torch.log(probs)
        loss = torch.sum(lb_one_hot * focal, dim=1)
        loss[ignore == 1] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid.sum()
        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'none':
            loss = loss.sum(dim=(1, 2)) / n_valid.float()
        return loss


class ClassificationFocalLoss(FocalLoss):

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C)
        args: label: tensor of shape (N, )
        '''
        with torch.no_grad():
            label = label.clone().detach().long()
            lb_one_hot = torch.zeros_like(logits).scatter_(
                1, label.unsqueeze(1), 1
            ).detach()
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[lb_one_hot == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(lb_one_hot == 1, probs, 1 - probs)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, lb_one_hot, reduce=False
        )
        loss = (alpha * torch.pow(1 - pt, self.gamma) * bce_loss).sum(dim=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()


class MultiFocalLoss(FocalLoss):
    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''

        with torch.no_grad():
            label = label.clone().detach().long()
            ignore = (label == self.ignore_lb)

            n_valid = (ignore == 0).sum(dim=(1, 2))
            label[ignore] = 0
            lb_one_hot = torch.zeros_like(logits).scatter_(
                1, label.unsqueeze(1), 1).detach()
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[lb_one_hot == 1] = self.alpha

        # compute loss
        probs = torch.sigmoid(logits)
        pt = torch.where(lb_one_hot == 1, probs, 1 - probs) + self.epsilon
        ce = - torch.log(pt)
        loss = (alpha * torch.pow(1. - pt, self.gamma) * ce).sum(dim=1)
        loss[ignore == 1] = 0

        if self.reduction == 'mean':
            loss = loss.sum() / n_valid.sum()
        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'none':
            loss = loss.sum(dim=(1, 2)) / n_valid.float()
        return loss
