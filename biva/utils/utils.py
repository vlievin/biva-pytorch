from typing import *

import torch
from torch.optim.lr_scheduler import _LRScheduler


def shp_cat(shps: List[Tuple[int]], dim: int):
    """concatenate tensor shapes"""
    out = list(shps[0])
    out[dim] = sum(list(s)[dim] for s in shps)
    return tuple(out)


def detach_to_device(x, device):
    """detach, clone and or place on the right device"""
    if x is not None:
        if isinstance(x, torch.Tensor):
            return x.detach().clone().to(device)
        else:
            return torch.tensor(x, device=device, dtype=torch.float)
    else:
        return None


def batch_reduce(x, reduce=torch.sum):
    batch_size = x.size(0)
    return reduce(x.view(batch_size, -1), dim=-1)


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum, eps: float = 1e-12, keepdim=False):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
    :param tensor: Tensor to compute LSE over
    :param dim: dimension to perform operation over
    :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
    :return: LSE
    """
    max, _ = torch.max(tensor, dim=dim, keepdim=keepdim)
    return torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=keepdim) + eps) + max


class LowerBoundedExponentialLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        lower_bound (float): lower bound for the learning rate.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, gamma, lower_bound, last_epoch=-1):
        self.gamma = gamma
        self.lower_bound = lower_bound
        super(LowerBoundedExponentialLR, self).__init__(optimizer, last_epoch)

    def _get_lr(self, base_lr):
        lr = base_lr * self.gamma ** self.last_epoch
        if lr < self.lower_bound:
            lr = self.lower_bound
        return lr

    def get_lr(self):
        return [self._get_lr(base_lr)
                for base_lr in self.base_lrs]
