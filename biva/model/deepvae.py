from typing import *

import torch
from torch import nn

from .architectures import get_deep_vae_mnist
from .stage import VaeStage, LvaeStage, BivaStage
from .utils import DataCollector
from ..layers import PaddedNormedConv

_default_enc, _default_z = get_deep_vae_mnist()


class DeepVae(nn.Module):
    """
    A Deep Hierarchical VAE.
    The model is a stack of N stages. Each stage features an inference and a generative path.
    Depending on the choice of the stage, multiple models can be implemented:
    - VAE: https://arxiv.org/abs/1312.6114
    - LVAE: https://arxiv.org/abs/1602.02282
    - BIVA: https://arxiv.org/abs/1902.02102
    """

    def __init__(self,
                 type: str = 'biva',
                 tensor_shp: Tuple[int] = (-1, 1, 28, 28),
                 stages: List[List[Tuple]] = _default_enc,
                 latents: List = _default_z,
                 nonlinearity: str = 'elu',
                 dropout: float = 0.,
                 features_out: Optional[int] = None,
                 lambda_init: Optional[Callable] = None,
                 **kwargs):

        """
        Initialize the Deep VAE model.

        :param type: model type (vae, lvae, biva)
        :param tensor_shp: Input tensor shape (batch_size, channels, *dimensions)
        :param stages: a list of list of tuple, each tuple describing a convolutional block (filters, stride, kernel_size)
        :param latents: a list describing the stochastic layers for each stage
        :param nonlinearity: activation function (gelu, elu, relu, tanh)
        :param dropout: dropout value
        :param features_out: optional number of output features if different from the input
        :param lambda_init: lambda function applied to the input
        :param kwargs: additional arugments passed to each stage
        """
        super().__init__()

        self.input_tensor_shape = tensor_shp
        self.lambda_init = lambda_init

        # select activation class
        Act = {'elu': nn.ELU, 'relu': nn.ReLU, 'tanh': nn.Tanh()}[nonlinearity]

        # seect stage class
        Stage = {'vae': VaeStage, 'lvae': LvaeStage, 'biva': BivaStage}[type]

        # build stages
        stages_ = []
        block_args = {'act': Act, 'dropout': dropout}

        input_shape = {'x': tensor_shp}
        for i, (conv_data, z_data) in enumerate(zip(stages, latents)):
            top_layer = i == len(stages) - 1
            bottom_layer = i == 0

            stage = Stage(input_shape, conv_data, z_data, top_layer, bottom_layer, **block_args, **kwargs)

            input_shape = stage.output_shape
            stages_ += [stage]

        self.stages = nn.ModuleList(stages_)

        # output convolution
        tensor_shp = self.stages[0].forward_shape
        if features_out is None:
            features_out = self.input_tensor_shape[1]
        conv_obj = nn.Conv2d if len(tensor_shp) == 4 else nn.Conv1d
        conv_out = conv_obj(tensor_shp[1], features_out, 1)
        conv_out = PaddedNormedConv(tensor_shp, conv_out, weightnorm=True)
        self.conv_out = nn.Sequential(Act(), conv_out)

    def infer(self, x: torch.Tensor, **kwargs: Any) -> List[Dict]:
        """
        Forward pass through the inference network and return the posterior of each layer order from the top to the bottom.
        :param x: input tensor
        :param kwargs: additional arguments passed to each stage
        :return: a list that contains the data for each stage
        """
        posteriors = []
        data = {'x': x}
        for stage in self.stages:
            data, posterior = stage.infer(data, **kwargs)
            posteriors += [posterior]

        return posteriors

    def generate(self, posteriors: Optional[List], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the generative model, compute KL and return reconstruction x_, KL and auxiliary data.
        If no posterior is provided, the prior is sampled.
        :param posteriors: a list containing the posterior for each stage
        :param kwargs: additional arguments passed to each stage
        :return: {'x_': reconstruction logits, 'kl': kl for each stage, **auxiliary}
        """
        if posteriors is None:
            posteriors = [None for _ in self.stages]

        output_data = DataCollector()
        x = None
        for posterior, stage in zip(posteriors[::-1], self.stages[::-1]):
            x, data = stage(x, posterior, **kwargs)
            output_data.extend(data)

        # output convolution
        x = self.conv_out(x)

        # sort data: [z1, z2, ..., z_L]
        output_data = output_data.sort()

        return {'x_': x, **output_data}

    def forward(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the inference model, the generative model and compute KL for each stage.
        x_ = p_\theta(x|z), z \sim q_\phi(z|x)
        kl_i = log q_\phi(z_i | h) - log p_\theta(z_i | h)

        :param x: input tensor
        :param kwargs: additional arguments passed to each stage
        :return: {'x_': reconstruction logits, 'kl': kl for each stage, **auxiliary}
        """

        if self.lambda_init is not None:
            x = self.lambda_init(x)

        posteriors = self.infer(x, **kwargs)

        data = self.generate(posteriors, N=x.size(0), **kwargs)

        return data

    def sample_from_prior(self, N: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """
        Sample the prior and pass through the generative model.
        x_ = p_\theta(x|z), z \sim p_\theta(z)

        :param N: number of samples (batch size)
        :param kwargs: additional arguments passed to each stage
        :return: {'x_': sample logits}
        """
        return self.generate(None, N=N, **kwargs)


class BIVA(DeepVae):
    def __init__(self, **kwargs):
        kwargs.pop('type', None)
        super().__init__(type='biva', **kwargs)


class LVAE(DeepVae):
    def __init__(self, **kwargs):
        kwargs.pop('type', None)
        super().__init__(type='lvae', **kwargs)


class VAE(DeepVae):
    def __init__(self, **kwargs):
        kwargs.pop('type', None)
        super().__init__(type='vae', **kwargs)
