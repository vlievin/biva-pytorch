from copy import copy
from typing import *

import torch
from torch import nn

from .stochastic import StochasticLayer
from .utils import DataCollector
from ..layers import GatedResNet
from ..utils import shp_cat


def StochasticBlock(data: Dict, *args, **kwargs):
    """Construct the stochastic block given by the Block argument"""
    Block = data.get('block')
    block = Block(data, *args, **kwargs)
    assert isinstance(block, StochasticLayer)
    return block


class DeterministicBlocks(nn.Module):

    def __init__(self,
                 tensor_shp: Tuple[int],
                 convolutions: List[Tuple[int]],
                 in_residual: bool = True,
                 transposed: bool = False,
                 Block: Any = GatedResNet,
                 aux_shape: Optional[Tuple[int]] = None,
                 **kwargs):
        """
        Defines a of sequence of deterministic blocks (resnets).
        You can extend this class by passing other Block classes as an argument.

        :param tensor_shp: input tensor shape as a tuple of integers (B, H, *D)
        :param convolutions: describes the sequence of blocks, each of them defined by a tuple  (filters, kernel_size, stride)
        :param aux_shape: auxiliary input tensor shape as a tuple of integers (B, H, *D)
        :param transposed: use transposed convolutions
        :param residual: use residual connections
        :param Block: Block object constructor (GatedResNet, ResMLP)
        """
        super().__init__()
        self.input_shape = tensor_shp
        layers = []
        for j, dim in enumerate(convolutions):
            residual = True if j > 0 else in_residual
            block = Block(tensor_shp, dim, aux_shape=aux_shape, transposed=transposed, residual=residual,
                          **kwargs)
            tensor_shp = block.output_shape
            layers += [block]

        self.layers = nn.ModuleList(layers)
        self.output_shape = tensor_shp

    def forward(self, x: torch.Tensor, aux: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        :param x: input tensor
        :param aux: auxiliary input tensor
        :return: output tensor
        """
        for layer in self.layers:
            x = layer(x, aux, **kwargs)
        return x


class VaeStage(nn.Module):
    def __init__(self,
                 input_shape: Dict[str, Tuple[int]],
                 convolutions: List[Tuple[int]],
                 stochastic: Tuple,
                 top_layer: bool,
                 bottom_layer: bool,
                 Block: Any = GatedResNet,
                 **kwargs):
        """
        VAE: https://arxiv.org/abs/1312.6114

        Define a Variational Autoencoder stage containing:
        - a sequence of convolutional blocks for the inference model
        - a sequence of convolutional blocks for the generative model
        - a stochastic layer

        :param input_shape: dictionary describing the input tensors of shapes (B, H, *D)
        :param convolution: list of tuple describing a convolutional block (filters, kernel_size, stride)
        :param stochastic: integer or tuple describing the stochastic layer: units or (units, kernel_size, discrete, K)
        :param top_layer: is top layer
        :param bottom_layer: is bottom layer
        :param Block: Block constructor
        :param kwargs: others arguments passed to the block constructors (both convolutions and stochastic)
        """
        super().__init__()

        tensor_shp = input_shape.get('x')
        aux_shape = input_shape.get('aux', None)

        self._input_shape = input_shape

        # define inference convolutional blocks
        in_residual = not bottom_layer
        self.q_convs = DeterministicBlocks(tensor_shp, convolutions, aux_shape=aux_shape, transposed=False,
                                           in_residual=in_residual, Block=Block, **kwargs)
        tensor_shp = self.q_convs.output_shape

        # define the stochastic layer
        self.stochastic = StochasticBlock(stochastic, tensor_shp, top_layer, **kwargs)
        z_shape = self.stochastic.output_shape

        # define the generative convolutional blocks
        aux_shape = None if top_layer else tensor_shp
        self.p_convs = DeterministicBlocks(z_shape, convolutions[::-1], aux_shape=aux_shape, transposed=True,
                                           in_residual=False, Block=Block, **kwargs)

        self._output_shape = {'x': z_shape, 'aux': tensor_shp}
        self._forward_shape = self.p_convs.output_shape

    @property
    def input_shape(self) -> Dict[str, Tuple[int]]:
        """size of the input tensors for the inference path"""
        return self._input_shape

    @property
    def output_shape(self) -> Dict[str, Tuple[int]]:
        """size of the output tensors for the inference path"""
        return self._output_shape

    @property
    def forward_shape(self) -> Tuple[int]:
        """size of the output tensor for the generative path"""
        return self._forward_shape

    def infer(self, data: Dict[str, torch.Tensor], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform a forward pass through the inference layers and sample the posterior.

        :param data: input data
        :param kwargs: additional parameters passed to the stochastic layer
        :return: (output data, variational data)
        """
        x = data.get('x')
        aux = data.get('aux', None)
        x = self.q_convs(x, aux)

        z, q_data = self.stochastic(x, inference=True, **kwargs)

        return {'x': z, 'aux': x}, q_data

    def forward(self, d: Optional[torch.Tensor], posterior: Optional[dict], **kwargs) -> Tuple[
        torch.Tensor, Dict[str, List]]:
        """
        Perform a forward pass through the generative model and compute KL if posterior data is available

        :param d: previous hidden state
        :param posterior: dictionary representing the posterior
        :return: (hidden state, dict('kl': [kl], **auxiliary) )
        """
        # sample p(z | d)
        z_p, p_data = self.stochastic(d, inference=False, **kwargs)

        # compute KL(q | p)
        if posterior is not None:
            loss_data = self.stochastic.loss(posterior, p_data, **kwargs)
            z = posterior.get('z')
        else:
            loss_data = {}
            z = z_p

        # pass through convolutions
        x = self.p_convs(z, aux=d)

        return x, loss_data


class LvaeStage(VaeStage):
    def __init__(self,
                 input_shape: Dict[str, Tuple[int]],
                 convolutions: List[Tuple[int]],
                 stochastic: Tuple,
                 top_layer: bool,
                 bottom_layer: bool,
                 Block: Any = GatedResNet,
                 dropout: float = 0,
                 **kwargs):
        """
        LVAE: https://arxiv.org/abs/1602.02282

        Define a Ladder Variational Autoencoder stage containing:
        - a sequence of convolutional blocks for the inference model
        - a sequence of convolutional blocks for the generative model
        - a stochastic layer

        :param input_shape: dictionary describing the input tensors of shapes (B, H, *D)
        :param convolution: list of tuple describing a convolutional block (filters, kernel_size, stride)
        :param stochastic: integer or tuple describing the stochastic layer: units or (units, kernel_size, discrete, K)
        :param top_layer: is top layer
        :param bottom_layer: is bottom layer
        :param kwargs: others arguments passed to the block constructors (both convolutions and stochastic)
        """
        super().__init__(input_shape, convolutions, stochastic, top_layer, bottom_layer, dropout=dropout, Block=Block,
                         **kwargs)

        # get the tensor shape of the output of the deterministic path
        top_shape = self._output_shape.get('aux')
        # modify the output of the inference path to be only deterministic
        self._output_shape = {'x': top_shape, 'aux': top_shape}

        aux_shape = top_shape if not top_layer else None
        conv = convolutions[-1]
        if isinstance(conv, list):
            conv = [conv[0], conv[1], 1, conv[-1]]
        self.merge = Block(top_shape, conv, aux_shape=aux_shape, transposed=False, in_residual=True, dropout=0,
                           **kwargs)

    def infer(self, data: Dict[str, torch.Tensor], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform a forward pass through the inference layers and sample the posterior.

        :param data: input data
        :param kwargs: additional parameters passed to the stochastic layer
        :return: (output data, variational data)
        """
        x = data.get('x')
        aux = data.get('aux', None)
        x = self.q_convs(x, aux)

        return {'x': x, 'aux': x}, {'h': x}

    def forward(self, d: Optional[torch.Tensor], posterior: Optional[dict], debugging: bool = False, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Perform a forward pass through the generative model and compute KL if posterior data is available

        :param d: previous hidden state
        :param posterior: dictionary representing the posterior
        :return: (hidden state, dict('kl': [kl]), **auxiliary)
        """

        # sample p(z | d)
        z_p, p_data = self.stochastic(d, inference=False, **kwargs)

        # sample q(z | h) and compute KL(q | p)
        if posterior is not None:
            # compute the top-down logits of q(z_i | x, z_{>i})
            h = posterior.get('h')
            h = self.merge(h, aux=d)

            # z ~ q(z | h_bu, d_td)
            z_q, q_data = self.stochastic(h, inference=True, **kwargs)

            loss_data = self.stochastic.loss(q_data, p_data, **kwargs)
            z = z_q
        else:
            loss_data = {}
            z = z_p

        # pass through convolutions
        x = self.p_convs(z, d)

        return x, loss_data


class BivaIntermediateStage(nn.Module):
    def __init__(self,
                 input_shape: Dict[str, Tuple[int]],
                 convolutions: List[Tuple[int]],
                 stochastic: Union[Dict, Tuple[Dict]],
                 top_layer: bool,
                 bottom_layer: bool,
                 dropout: float = 0,
                 conditional_bu: bool = False,
                 Block: Any = GatedResNet,
                 **kwargs):
        """
        BIVA: https://arxiv.org/abs/1902.02102

        Define a Bidirectional Variational Autoencoder stage containing:
        - a sequence of convolutional blocks for the bottom-up inference model (BU)
        - a sequence of convolutional blocks for the top-down inference model (TD)
        - a sequence of convolutional blocks for the generative model
        - two stochastic layers (BU and TD)

        :param input_shape: dictionary describing the input tensor shape (B, H, *D)
        :param convolution: list of tuple describing a convolutional block (filters, kernel_size, stride)
        :param stochastic: dictionary describing the stochastic layer: units or (units, kernel_size, discrete, K)
        :param top_layer: is top layer
        :param bottom_layer: is bottom layer
        :param dropout: dropout value
        :param conditional_bu: condition BU prior on p(z_TD)
        :param aux_shape: auxiliary input tensor shape as a tuple of integers (B, H, *D)
        :param kwargs: others arguments passed to the block constructors (both convolutions and stochastic)
        """
        super().__init__()

        if 'x' in input_shape.keys():
            bu_shp = td_shp = input_shape.get('x')
            aux_shape = None
        else:
            bu_shp = input_shape.get('x_bu')
            td_shp = input_shape.get('x_td')
            aux_shape = input_shape.get('aux')

        if isinstance(stochastic.get('block'), tuple):
            bu_block, td_block = stochastic.get('block')
            bu_stochastic = copy(stochastic)
            td_stochastic = copy(stochastic)
            bu_stochastic['block'] = bu_block
            td_stochastic['block'] = td_block
        else:
            bu_stochastic = td_stochastic = stochastic

        self._input_shape = input_shape

        # define inference convolutional blocks
        in_residual = not bottom_layer
        self.q_bu_convs = DeterministicBlocks(bu_shp, convolutions, aux_shape=aux_shape, transposed=False,
                                              in_residual=in_residual, dropout=dropout, Block=Block, **kwargs)
        aux_shape = self.q_bu_convs.output_shape
        self.q_td_convs = DeterministicBlocks(td_shp, convolutions, aux_shape=aux_shape, transposed=False,
                                              in_residual=in_residual, dropout=dropout, Block=Block, **kwargs)

        # shape of the output of the inference path and input tensor from the generative path
        top_tensor_shp = self.q_td_convs.output_shape

        # define the BU stochastic layer
        bu_top = False if conditional_bu else top_layer
        self.bu_stochastic = StochasticBlock(bu_stochastic, top_tensor_shp, bu_top, **kwargs)

        # define the TD stochastic layer
        self.td_stochastic = StochasticBlock(td_stochastic, top_tensor_shp, top_layer, **kwargs)

        # output shape
        z_shape = self.bu_stochastic.output_shape

        # with its merge layer
        aux_shape = top_tensor_shp if not top_layer else None
        conv = convolutions[-1]
        if isinstance(conv, list):
            conv = [conv[0], conv[1], 1, conv[-1]]
        self.merge = Block(top_tensor_shp, conv, aux_shape=aux_shape, transposed=False, in_residual=True, dropout=0,
                           **kwargs)

        # define the condition p(z_bu | z_td, ...)
        if conditional_bu:
            self.bu_condition = Block(z_shape, conv, aux_shape=aux_shape, transposed=False, in_residual=False,
                                      dropout=0, **kwargs)
        else:
            self.bu_condition = None

        # define the generative convolutional blocks
        aux_shape = None if top_layer else top_tensor_shp
        p_in_shp = shp_cat([z_shape, z_shape], 1)
        self.p_convs = DeterministicBlocks(p_in_shp, convolutions[::-1], aux_shape=aux_shape, transposed=True,
                                           in_residual=False, dropout=dropout, Block=Block, **kwargs)

        self._output_shape = {'x_bu': z_shape,
                              'x_td': top_tensor_shp,
                              'aux': shp_cat([top_tensor_shp, top_tensor_shp], 1)}

        self._forward_shape = self.p_convs.output_shape

    @property
    def input_shape(self) -> Dict[str, Tuple[int]]:
        """size of the input tensors for the inference path"""
        return self._input_shape

    @property
    def output_shape(self) -> Dict[str, Tuple[int]]:
        """size of the output tensors for the inference path"""
        return self._output_shape

    @property
    def forward_shape(self) -> Tuple[int]:
        """size of the output tensor for the generative path"""
        return self._forward_shape

    def infer(self, data: Dict[str, torch.Tensor], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform a forward pass through the inference layers and sample the posterior.

        :param data: input data
        :param kwargs: additional parameters passed to the stochastic layer
        :return: (output data, variational data)
        """
        if 'x' in data.keys():
            x = data.get('x')
            x_bu, x_td = x, x
        else:
            x_bu = data.get('x_bu')
            x_td = data.get('x_td')

        aux = data.get('aux', None)

        # BU path
        x_bu = self.q_bu_convs(x_bu, aux=aux)
        # z_bu ~ q(x)
        z_bu, bu_q_data = self.bu_stochastic(x_bu, inference=True, **kwargs)

        # TD path
        x_td = self.q_td_convs(x_td, aux=x_bu)
        td_q_data = {'z': z_bu, 'h': x_td}  # note h = d_q(x)

        # skip connection
        aux = torch.cat([x_bu, x_td], 1)

        return {'x_bu': z_bu, 'x_td': x_td, 'aux': aux}, {'z_bu': z_bu, 'bu': bu_q_data, 'td': td_q_data}

    def forward(self, d: Optional[torch.Tensor], posterior: Optional[dict], debugging: bool = False, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Perform a forward pass through the generative model and compute KL if posterior data is available

        :param d: previous hidden state
        :param posterior: dictionary representing the posterior
        :return: (hidden state, dict('kl': [kl], **auxiliary))
        """

        if posterior is not None:
            # sample posterior and compute KL using prior
            bu_q_data = posterior.get('bu')
            td_q_data = posterior.get('td')
            z_bu_q = posterior.get('z_bu')

            # top-down: compute the posterior using the bottom-up hidden state and top-down hidden state
            # z_td ~ p(d_top)
            z_td_p, td_p_data = self.td_stochastic(d, inference=False, **kwargs)

            # merge d_top with h = d_q(x)
            h = td_q_data.get('h')
            h = self.merge(h, aux=d)

            # z_td ~ q(z | h)
            z_td_q, td_q_data = self.td_stochastic(h, inference=True, **kwargs)

            # compute log q_bu(z_i | x) - log p_bu(z_i) (+ additional data)
            td_loss_data = self.td_stochastic.loss(td_q_data, td_p_data, **kwargs)

            # conditional BU
            if self.bu_condition is not None:
                d_ = self.bu_condition(z_td_q, aux=d)
            else:
                d_ = d

            # bottom-up: retrieve data from the inference path
            # z_bu ~ p(d_top)
            z_bu_p, bu_p_data = self.bu_stochastic(d_, inference=False, **kwargs)

            # compute log q_td(z_i | x, z_{>i}) - log p_td(z_i) (+ additional data)
            bu_loss_data = self.bu_stochastic.loss(bu_q_data, bu_p_data, **kwargs)

            # merge samples
            z = torch.cat([z_td_q, z_bu_q], 1)

        else:
            # sample priors
            # top-down
            z_td_p, td_p_data = self.td_stochastic(d, inference=False, **kwargs)  # prior

            # conditional BU
            if self.bu_condition is not None:
                d_ = self.bu_condition(z_td_p, aux=d)
            else:
                d_ = d

            # bottom-up
            z_bu_p, bu_p_data = self.bu_stochastic(d_, inference=False, **kwargs)  # prior

            bu_loss_data, td_loss_data = {}, {}
            z = torch.cat([z_bu_p, z_td_p], 1)

        # pass through convolutions
        x = self.p_convs(z, aux=d)

        # gather data
        loss_data = DataCollector()
        loss_data.extend(td_loss_data)
        loss_data.extend(bu_loss_data)

        return x, loss_data


class BivaTopStage(VaeStage):
    def __init__(self, input_shape: Dict[str, Tuple[int]], *args, **kwargs):
        bu_shp = input_shape.get('x_bu')
        td_shp = input_shape.get('x_td')

        tensor_shp = shp_cat([bu_shp, td_shp], 1)
        concat_shape = {'x': tensor_shp}

        super().__init__(concat_shape, *args, **kwargs)

    def infer(self, data: Dict[str, torch.Tensor], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        x_bu = data.pop('x_bu')
        x_td = data.pop('x_td')
        data['x'] = torch.cat([x_bu, x_td], 1)

        return super().infer(data, **kwargs)


def BivaStage(input_shape: Dict[str, Tuple[int]],
              convolutions: List[Tuple[int]],
              stochastic: Union[Dict, Tuple[Dict]],
              top_layer: bool,
              bottom_layer: bool,
              **kwargs):
    """
    BIVA: https://arxiv.org/abs/1902.02102

    Define a Bidirectional Variational Autoencoder stage containing:
    - a sequence of convolutional blocks for the bottom-up inference model (BU)
    - a sequence of convolutional blocks for the top-down inference model (TD)
    - a sequence of convolutional blocks for the generative model
    - two stochastic layers (BU and TD)

    :param input_shape: dictionary describing the input tensor shape (B, H, *D)
    :param convolution: list of tuple describing a convolutional block (filters, kernel_size, stride)
    :param stochastic: dictionary describing the stochastic layer: units or (units, kernel_size, discrete, K)
    :param top_layer: is top layer
    :param bottom_layer: is bottom layer
    :param dropout: dropout value
    :param conditional_bu: condition BU prior on p(z_TD)
    :param aux_shape: auxiliary input tensor shape as a tuple of integers (B, H, *D)
    :param kwargs: others arguments passed to the block constructors (both convolutions and stochastic)
    """

    if top_layer:
        return BivaTopStage(input_shape, convolutions, stochastic, top_layer, bottom_layer, **kwargs)
    else:
        return BivaIntermediateStage(input_shape, convolutions, stochastic, top_layer, bottom_layer, **kwargs)
