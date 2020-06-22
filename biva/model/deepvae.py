from typing import *

import torch
from torch import nn

from .architectures import get_deep_vae_mnist
from .stage import VaeStage, LvaeStage, BivaStage
from .utils import DataCollector
from ..layers import PaddedNormedConv


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
                 Stage: Any = BivaStage,
                 tensor_shp: Tuple[int] = (-1, 1, 28, 28),
                 padded_shp: Optional[Tuple] = None,
                 stages: List[List[Tuple]] = None,
                 latents: List = None,
                 nonlinearity: str = 'elu',
                 q_dropout: float = 0.,
                 p_dropout: float = 0.,
                 features_out: Optional[int] = None,
                 lambda_init: Optional[Callable] = None,
                 projection: Optional[nn.Module] = None,
                 **kwargs):

        """
        Initialize the Deep VAE model.
        :param Stage: stage constructor (VaeStage, LvaeStage, BivaStage)
        :param tensor_shp: Input tensor shape (batch_size, channels, *dimensions)
        :param padded_shp: pad input tensor to this shape
        :param stages: a list of list of tuple, each tuple describing a convolutional block (filters, stride, kernel_size)
        :param latents: a list describing the stochastic layers for each stage
        :param nonlinearity: activation function (gelu, elu, relu, tanh)
        :param q_dropout: inference dropout value
        :param p_dropout: generative dropout value
        :param features_out: optional number of output features if different from the input
        :param lambda_init: lambda function applied to the input
        :param projection: projection layer with constructor __init__(output_shape)

        :param kwargs: additional arugments passed to each stage
        """
        super().__init__()
        stages, latents = self.get_default_architecture(stages, latents)

        self.input_tensor_shape = tensor_shp
        self.lambda_init = lambda_init

        # input padding
        if padded_shp is not None:
            padding = [[(t - o) // 2, (t - o) // 2] for t, o in zip(padded_shp, tensor_shp[2:])]
            self.pad = [u for pads in padding for u in pads]
            self.unpad = [-u for u in self.pad]
            in_shp = [*tensor_shp[:2], *padded_shp]
        else:
            self.pad = None
            in_shp = tensor_shp

        # select activation class
        Act = {'elu': nn.ELU, 'relu': nn.ReLU, 'tanh': nn.Tanh()}[nonlinearity]

        # initialize the inference path
        stages_ = []
        block_args = {'act': Act, 'q_dropout': q_dropout, 'p_dropout': p_dropout}

        input_shape = {'x': in_shp}
        for i, (conv_data, z_data) in enumerate(zip(stages, latents)):
            top = i == len(stages) - 1
            bottom = i == 0

            stage = Stage(input_shape, conv_data, z_data, top=top, bottom=bottom, **block_args, **kwargs)

            input_shape = stage.q_output_shape
            stages_ += [stage]

        self.stages = nn.ModuleList(stages_)

        if projection is None:
            # output convolution
            tensor_shp = self.stages[0].p_output_shape['d']
            if features_out is None:
                features_out = self.input_tensor_shape[1]
            conv_obj = nn.Conv2d if len(tensor_shp) == 4 else nn.Conv1d
            conv_out = conv_obj(tensor_shp[1], features_out, 1)
            conv_out = PaddedNormedConv(tensor_shp, conv_out, weightnorm=True)
            self.projection = nn.Sequential(Act(), conv_out)
        else:
            tensor_shp = self.stages[0].forward_shape['d']
            self.projection = projection(tensor_shp)

    def get_default_architecture(self, stages, latents):
        if stages is None:
            stages, _ = get_deep_vae_mnist()

        if latents is None:
            _, latens = get_deep_vae_mnist()

        return stages, latents

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
        x = {}
        for posterior, stage in zip(posteriors[::-1], self.stages[::-1]):
            x, data = stage(x, posterior, **kwargs)
            output_data.extend(data)

        # output convolution
        x = self.projection(x['d'])

        # undo padding
        if self.pad is not None:
            x = nn.functional.pad(x, self.unpad)

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

        if self.pad is not None:
            x = nn.functional.pad(x, self.pad)

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
        kwargs.pop('Stage', None)
        super().__init__(Stage=BivaStage, **kwargs)


class LVAE(DeepVae):
    def __init__(self, **kwargs):
        kwargs.pop('Stage', None)
        super().__init__(Stage=LvaeStage, **kwargs)


class VAE(DeepVae):
    def __init__(self, **kwargs):
        kwargs.pop('Stage', None)
        super().__init__(Stage=VaeStage, **kwargs)
