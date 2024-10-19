import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def bce_with_logits_loss1(input, target, weight=None, pos_weight=None, reduce=False):
    max_val = torch.maximum(-input,torch.zeros(1).cuda())

    if pos_weight is not None:
        log_weight = ((pos_weight - 1) * target) + 1
        loss = (1 - target) * input
        loss_1 = torch.log(torch.exp(-max_val) + torch.exp(-input - max_val)) + max_val
        loss += log_weight * loss_1
    else:
        loss = (1 - target) * input
        loss += max_val
        loss += torch.log(torch.exp(-max_val) + torch.exp(-input - max_val))

    if weight is not None:
        loss = loss * weight
    if reduce is False:
        return loss
    else:
        return torch.mean(loss)
class PML_Confidence(nn.Module):
    def __init__(self, train_givenY, mu=0.1):
        super().__init__()
        self.mu = mu
        print('Calculating uniform targets...')
        # calculate confidence
        self.confidence = train_givenY.float()/train_givenY.sum(dim=1, keepdim=True)
        self.distribution = self.confidence.sum(0)/self.confidence.sum()

    def forward(self, logits, index, targets=None):
        if targets is None:
            loss_vec = bce_with_logits_loss1(logits,self.confidence[index, :].detach(), reduce=False)
        else:
            loss_vec = bce_with_logits_loss1(logits, targets, reduce=False)
        average_loss = loss_vec.mean()
        return average_loss, loss_vec

    @torch.no_grad()
    def get_distribution(self):
        self.update_distribution()
        return self.distribution

    @torch.no_grad()
    def update_distribution(self):
        self.distribution = self.confidence.sum(0) / self.confidence.sum()

    @torch.no_grad()
    def confidence_move_update(self, temp_un_conf, batch_index, ratio=None):
        if ratio:
            self.confidence[batch_index, :] = self.confidence[batch_index, :] * (1 - ratio) + temp_un_conf * ratio
        else:
            self.confidence[batch_index, :] = self.confidence[batch_index, :] * (1 - self.mu) + temp_un_conf * self.mu
        return None


