import math
from functools import partial
from typing import *

import numpy as np
import torch
from torch import Tensor, nn

from .freebits import FreeBits
from ..utils import batch_reduce, log_sum_exp, detach_to_device


class VariationalInference(object):
    def __init__(self, likelihood: any, iw_samples: int = 1, auxiliary: Dict[str, float] = {}, **parameters: Any):
        """
        Initialize VI evaluator
        :param likelihood: likelihood class used to evaluate log p(x | z)
        :param iw_samples: number of importance-weighted samples
        :param auxiliary: dict of auxiliary losses, each key must have a match in the model output, the auxiliary values define the weight of the auxiliary loss in the overall loss
        :param parameters: additional parameters passed to model and evaluator
        """
        super().__init__()

        assert iw_samples > 0
        self._iw_samples = iw_samples
        self._parameters = parameters
        self._auxiliary = auxiliary
        self.likelihood = likelihood

    @staticmethod
    def compute_kls(kls: Union[Tensor, List[Tensor]], freebits: Optional[Union[float, List[float]]], device: str):
        """compute kl and kl to be accounted in the loss"""

        if kls is None or (isinstance(kls, list) and len(kls) == 0):
            _zero = detach_to_device(0., device)
            return _zero, _zero

        # set kls and freebits as lists
        if not isinstance(kls, list):
            kls = [kls]
            if freebits is not None and not isinstance(freebits, list):
                freebits = [freebits]

        # apply freebits to each
        if freebits is not None:
            kls_loss = (FreeBits(fb)(kl) for fb, kl in zip(freebits, kls))
        else:
            kls_loss = kls

        # sum freebit kls
        kls_loss = [batch_reduce(kl)[:, None] for kl in kls_loss]
        kls_loss = batch_reduce(torch.cat(kls_loss, 1))

        # sum kls
        kls = [batch_reduce(kl)[:, None] for kl in kls]
        kls = batch_reduce(torch.cat(kls, 1))

        return kls, kls_loss

    def compute_elbo(self, x, outputs, beta=1.0, freebits=0, **kwargs):

        # Destructuring dict
        x_ = outputs.get('x_')
        kls = outputs.get('kl')

        # compute E_p(x) [ - log p_\theta(x | z) ]
        nll = - batch_reduce(self.likelihood(logits=x_).log_prob(x))

        # compute kl: \sum_i E_q(z_i) [ log q(z_i | h) - log p(z_i | h) ]
        kl, kls_loss = self.compute_kls(kls, freebits, x.device)

        # compute total loss and elbo
        loss = nll + beta * kls_loss
        elbo = -(nll + kl)

        # compute auxiliary losses / kls
        auxiliary = {}
        for k, default_value in self._auxiliary.items():
            # compute value
            value = outputs.get(k, None)
            value, _ = self.compute_kls(value, None, x.device)

            # get custom weights from kwargs
            weight = kwargs.get(k, default_value)

            # add to loss
            loss = loss + weight * value

            # store as a tuple
            auxiliary[k] = (weight, value)

        return loss, elbo, kls, kl, nll, auxiliary

    def __call__(self, model: nn.Module, x: Tensor, **kwargs: Any) -> Tuple[Tensor, Dict]:
        """
        Process inputs using model and compute loss, ELBO and diagnostics.
        :param model: model to evaluate
        :param x: input tensor
        :param kwargs: other args passed both to the model and the evaluator
        :return: (loss, diagnostics)
        """

        # update kwargs
        kwargs.update(self._parameters)

        # importance-weighted placeholders
        iw_elbos = torch.zeros((self._iw_samples, x.size(0)), device=x.device, dtype=torch.float)
        iw_kls = torch.zeros((self._iw_samples, x.size(0)), device=x.device, dtype=torch.float)
        iw_nlls = torch.zeros((self._iw_samples, x.size(0)), device=x.device, dtype=torch.float)

        # Effective Sample size (requires KL to be computed as an estimate)
        ratios = torch.zeros((self._iw_samples, x.size(0)), device=x.device, dtype=torch.float)

        # feed forward pass
        for k in range(self._iw_samples):
            # forward pass
            outputs = model(x, **kwargs)

            # compute VI elbo
            loss, elbo, kls, kl, nll, auxiliary = self.compute_elbo(x, outputs, **kwargs)
            iw_elbos[k, :] = elbo
            iw_kls[k, :] = - kl
            iw_nlls[k, :] = - nll
            ratios[k, :] = torch.exp(-kl)

        if self._iw_samples > 1:
            elbo = log_sum_exp(iw_elbos, dim=0, sum_op=torch.mean)
            kl = - log_sum_exp(iw_kls, dim=0, sum_op=torch.mean)
            nll = - log_sum_exp(iw_nlls, dim=0, sum_op=torch.mean)

        # Compute effective sample size
        N_eff = torch.sum(ratios, 0) ** 2 / torch.sum(ratios ** 2, 0)

        # gather diagnostics
        bits_per_dim = elbo / math.log(2.) / np.prod(x.size()[1:])
        format = partial(detach_to_device, device=x.device)
        diagnostics = {
            "loss": {"loss": format(loss), "elbo": format(elbo), "kl": format(kl), "nll": format(nll),
                     "bpd": format(bits_per_dim)},
            "info": {"N_eff": format(N_eff), "batch_size": x.size(0)}
        }

        return loss.mean(), diagnostics
