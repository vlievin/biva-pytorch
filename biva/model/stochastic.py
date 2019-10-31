from typing import *

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.distributions import Normal

from ..layers import PaddedNormedConv, NormedDense, NormedLinear
from ..utils import batch_reduce


class StochasticLayer(nn.Module):
    """
    An abstract class of a VAE stochastic layer.
    """

    def __init__(self, data: Dict, tensor_shp: Tuple[int], **kwargs: Any):
        super().__init__()
        pass

    def forward(self, x: Optional[torch.Tensor], inference: bool, N: Optional[int] = None, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, Any]]:
        """
        Sample the stochastic layer, if no hidden state is provided, uses the prior.
        :param x: hidden state used to computed logits (Optional : None means using the prior)
        :param inference: inference mode
        :param N: number of samples (when sampling from prior)
        :param kwargs: additional args passed ot the stochastic layer
        :return: (projected sample, data)
        """
        raise NotImplementedError

    def loss(self, q_data: Dict[str, Any], p_data: Dict[str, Any], **kwargs: Any) -> Dict[str, List[Any]]:
        """
        Compute the KL divergence and other auxiliary losses if required
        :param q_data: data received from the posterior forward pass
        :param p_data: data received from the prior forward pass
        :param kwargs: other parameters passed to the kl function
        :return: dictionary of losses {'kl': [values], 'auxiliary' : [aux_values], ...}
        """
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError


class DenseNormal(StochasticLayer):
    """
    A Normal stochastic layer parametrized by dense layers.
    """

    def __init__(self, data: Tuple, tensor_shp: Tuple[int], top_layer: bool, act: nn.Module = nn.ELU,
                 weightnorm: bool = True, **kwargs):
        super().__init__(data, tensor_shp)

        self._output_shape = tensor_shp
        self._input_shape = tensor_shp

        nhid = tensor_shp[1]
        self.nz = data.get('N')
        self.tensor_shp = tensor_shp
        self.act = act()
        self.dim = 2

        # stochastic layer and prior
        if top_layer:
            prior = torch.zeros((2 * self.nz))
            self.register_buffer('prior', prior)

        # computes logits
        nz_in = 2 * self.nz
        self.px2z = NormedDense(tensor_shp, nz_in, weightnorm=weightnorm)
        self.qx2z = NormedDense(tensor_shp, nz_in, weightnorm=weightnorm)

        # project sample back to the original shape
        nz_out = self.nz
        out_shp = np.prod([nhid, *tensor_shp[2:]])
        self.z_proj = NormedLinear(nz_out, out_shp, weightnorm=weightnorm)

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def input_shape(self):
        return self._input_shape

    def compute_logits(self, x: torch.Tensor, inference: bool) -> Tensor:
        """
        Compute the logits of the distribution.
        :param x: input tensor
        :param inference: inference mode
        :return: logits
        """
        x = self.act(x)

        if inference:
            logits = self.qx2z(x)
        else:
            logits = self.px2z(x)

        # apply activation to logvar
        mu, logvar = logits.chunk(2, dim=1)
        logvar = self.act(logvar)
        return mu, logvar

    def forward(self, x: Optional[Tensor], inference: bool, N: Optional[int] = None, **kwargs) -> Tuple[
        Tensor, Dict[str, Any]]:

        if x is None:
            mu, logvar = self.prior.expand(N, *self.prior.shape).chunk(2, dim=1)
        else:
            mu, logvar = self.compute_logits(x, inference)

        # sample layer
        std = logvar.mul(0.5).exp()
        dist = Normal(mu, std)
        z_ = dist.rsample()

        # project back to hidden state space
        z = self.z_proj(z_)
        z = z.view(self.tensor_shp)

        return z, {'z': z, 'z_': z_, 'dist': dist}

    def loss(self, q_data: Dict[str, Any], p_data: Dict[str, Any], **kwargs: Any) -> Dict[str, List]:
        z_q = q_data.get('z_')
        q = q_data.get('dist')
        p = p_data.get('dist')

        kl = q.log_prob(z_q) - p.log_prob(z_q)
        kl = batch_reduce(kl)

        return {'kl': [kl]}


class ConvNormal(StochasticLayer):
    """
    A Normal stochastic layer parametrized by convolutions.
    """

    def __init__(self, data: Dict, tensor_shp: Tuple[int], top_layer: bool, act: nn.Module = nn.ELU,
                 learn_prior: bool = True, **kwargs):
        super().__init__(data, tensor_shp)

        nhid = tensor_shp[1]
        self.nz = data.get('N')
        kernel_size = data.get('kernel')
        self.tensor_shp = tensor_shp
        self.input_shp = tensor_shp
        self.act = act()

        # prior
        if top_layer:
            prior = torch.zeros((2 * self.nz, *tensor_shp[2:]))

            if learn_prior:
                self.prior = nn.Parameter(prior)
            else:
                self.register_buffer('prior', prior)

        # computes logits
        nz_in = 2 * self.nz
        self.px2z = PaddedNormedConv(tensor_shp, nn.Conv2d(nhid, nz_in, kernel_size))
        self.qx2z = PaddedNormedConv(tensor_shp, nn.Conv2d(nhid, nz_in, kernel_size))

        # compute output shape
        nz_out = self.nz
        out_shp = (-1, nz_out, *tensor_shp[2:])
        self._output_shape = out_shp
        self._input_shape = tensor_shp

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def input_shape(self):
        return self._input_shape

    def compute_logits(self, x: torch.Tensor, inference: bool) -> torch.Tensor:
        """
        Compute the logits of the distribution.
        :param x: input tensor
        :param inference: inference mode
        :return: logits
        """
        x = self.act(x)
        if inference:
            logits = self.qx2z(x)
        else:
            logits = self.px2z(x)

        # apply activation to logvar
        mu, logvar = logits.chunk(2, dim=1)
        return mu, self.act(logvar)

    def expand_prior(self, batch_size: int):
        return self.prior.expand(batch_size, *self.prior.shape).chunk(2, dim=1)

    def forward(self, x: Optional[torch.Tensor], inference: bool, N: Optional[int] = None, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, Any]]:

        if x is None:
            mu, logvar = self.expand_prior(N)
        else:
            mu, logvar = self.compute_logits(x, inference)

        # sample layer
        std = logvar.mul(0.5).exp()
        dist = Normal(mu, std)
        z_ = dist.rsample()

        # project back (here, no projection)
        z = z_

        return z, {'z': z, 'z_': z_, 'dist': dist}

    def loss(self, q_data: Dict[str, Any], p_data: Dict[str, Any], **kwargs: Any) -> Dict[str, List]:
        z_q = q_data.get('z_')
        q = q_data.get('dist')
        p = p_data.get('dist')

        kl = q.log_prob(z_q) - p.log_prob(z_q)
        kl = batch_reduce(kl)

        return {'kl': [kl]}
