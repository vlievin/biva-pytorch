import numpy as np
import torch


class FreeBits():
    """
    free bits: https://arxiv.org/abs/1606.04934
    Assumes a each of the dimension to be one group
    """

    def __init__(self, min_KL: float):
        self.min_KL = min_KL

    def __call__(self, kls: torch.Tensor) -> torch.Tensor:
        """
        Apply freebits over tensor. The freebits budget is distributed equally among dimensions.
        The returned freebits KL is equal to max(kl, freebits_per_dim, dim = >0)
        :param kls: KL of shape [batch size, *dimensions]
        :return:  freebits KL of shape [batch size, *dimensions]
        """

        # equally divide freebits budget over the dimensions
        dimensions = np.prod(kls.shape[1:])
        min_KL_per_dim = self.min_KL / dimensions if len(kls.shape) > 1 else self.min_KL
        min_KL_per_dim = min_KL_per_dim * torch.ones_like(kls)

        # apply freebits
        freebits_kl = torch.cat([kls.unsqueeze(-1), min_KL_per_dim.unsqueeze(-1)], -1)
        freebits_kl = torch.max(freebits_kl, dim=-1)[0]

        return freebits_kl
