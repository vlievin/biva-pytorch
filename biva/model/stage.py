from copy import copy
from typing import *

import torch
from torch import nn, Tensor

from .stochastic import StochasticLayer
from .utils import DataCollector
from ..layers import GatedResNet, AsFeatureMap
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
                 aux_shape: Optional[List[Tuple[int]]] = None,
                 **kwargs):
        """
        Defines a of sequence of deterministic blocks (resnets).
        You can extend this class by passing other Block classes as an argument.

        auxiliary connections: if the number of auxiliary inputs is smaller than the number of layers,
        the auxiliary inputs are repeated to match the number of layers.

        :param tensor_shp: input tensor shape as a tuple of integers (B, H, *D)
        :param convolutions: describes the sequence of blocks, each of them defined by a tuple  (filters, kernel_size, stride)
        :param aux_shape: auxiliary input tensor shape as a tuple of integers (B, H, *D)
        :param transposed: use transposed convolutions
        :param residual: use residual connections
        :param Block: Block object constructor (GatedResNet, ResMLP)
        """
        super().__init__()
        self.input_shape = tensor_shp
        self._use_skips = True
        layers = []

        if aux_shape is None:
            self._use_skips = False
            aux_shape = []

        for j, dim in enumerate(convolutions):
            residual = True if j > 0 else in_residual
            aux = aux_shape.pop() if self._use_skips else None
            block = Block(tensor_shp, dim, aux_shape=aux, transposed=transposed, residual=residual,
                          **kwargs)
            tensor_shp = block.output_shape
            aux_shape = [tensor_shp] + aux_shape
            layers += [block]

        self.layers = nn.ModuleList(layers)
        self.output_shape = tensor_shp
        self.hidden_shapes = aux_shape

    def __len__(self):
        return len(self.layers)

    def forward(self, x: Tensor, aux: Optional[List[Tensor]] = None, **kwargs) -> Tuple[Tensor, List[Tensor]]:
        """
        :param x: input tensor
        :param aux: list of auxiliary inputs
        :return: output tensor, activations
        """
        if aux is None:
            aux = []

        for layer in self.layers:
            a = aux.pop() if self._use_skips else None
            x = layer(x, a, **kwargs)
            aux = [x] + aux

        return x, aux


class BaseStage(nn.Module):
    def __init__(self,
                 input_shape: Dict[str, Tuple[int]],
                 convolutions: List[Tuple[int]],
                 stochastic: Tuple,
                 top: bool = False,
                 bottom: bool = False,
                 q_dropout: float = 0,
                 p_dropout: float = 0,
                 Block: Any = GatedResNet,
                 no_skip: bool = False,
                 **kwargs):
        """
        Define a stage of a hierarchical model.
        In a VAE setting, a stage defines:
        * the latent variable z_i
        * the encoder q(z_i | h_{q<i})
        * the decoder p(z_{i-1} | z_i)
        """
        super().__init__()
        self._input_shape = input_shape
        self._convolutions = convolutions
        self._stochastic = stochastic
        self._top = top
        self._bottom = bottom
        self._no_skip = no_skip

    @property
    def input_shape(self) -> Dict[str, Tuple[int]]:
        """size of the input tensors for the inference path"""
        return self._input_shape

    @property
    def q_output_shape(self) -> Dict[str, Tuple[int]]:
        """size of the output tensors for the inference path"""
        raise NotImplementedError

    @property
    def forward_shape(self) -> Tuple[int]:
        """size of the output tensor for the generative path"""
        raise NotImplementedError

    def infer(self, data: Dict[str, Tensor], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform a forward pass through the inference layers and sample the posterior.

        :param data: input data
        :param kwargs: additional parameters passed to the stochastic layer
        :return: (output data, variational data)
        """
        raise NotImplementedError

    def forward(self, data: dict, posterior: Optional[dict], **kwargs) -> Tuple[
        Dict, Dict[str, List]]:
        """
        Perform a forward pass through the generative model and compute KL if posterior data is available

        :param data: data from the above stage forward pass
        :param posterior: dictionary representing the posterior from same stage inference pass
        :return: (dict('d' : d, 'aux : [aux]), dict('kl': [kl], **auxiliary) )
        """
        raise NotImplementedError


class VaeStage(BaseStage):
    def __init__(self,
                 input_shape: Dict[str, Tuple[int]],
                 convolutions: List[Tuple[int]],
                 stochastic: Tuple,
                 top: bool = False,
                 bottom: bool = False,
                 q_dropout: float = 0,
                 p_dropout: float = 0,
                 Block: Any = GatedResNet,
                 no_skip: bool = False,
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
        :param top: is top layer
        :param bottom: is bottom layer
        :param q_dropout: inference dropout value
        :param p_dropout: generative dropout value
        :param Block: Block constructor
        :param no_skip: do not use skip connections
        :param kwargs: others arguments passed to the block constructors (both convolutions and stochastic)
        """
        super().__init__(input_shape, convolutions, stochastic, top=top, bottom=bottom, q_dropout=q_dropout,
                         p_dropout=p_dropout, Block=Block, no_skip=no_skip)

        tensor_shp = input_shape.get('x')
        aux_shape = input_shape.get('aux', None)

        # mute skip connections
        if no_skip:
            aux_shape = None

        # define inference convolutional blocks
        in_residual = not bottom
        q_skips = [aux_shape for _ in convolutions] if aux_shape is not None else None
        self.q_convs = DeterministicBlocks(tensor_shp, convolutions, aux_shape=q_skips, transposed=False,
                                           in_residual=in_residual, Block=Block, dropout=q_dropout, **kwargs)

        # shape of the deterministic output
        tensor_shp = self.q_convs.output_shape

        # define the stochastic layer
        self.stochastic = StochasticBlock(stochastic, tensor_shp, top=top, **kwargs)
        self.q_proj = AsFeatureMap(self.stochastic.output_shape, self.stochastic.input_shape)

        self._q_output_shape = {'x': self.q_proj.output_shape, 'aux': tensor_shp}

        ### GENERATIVE MODEL

        # project z sample
        self.p_proj = AsFeatureMap(self.stochastic.output_shape, self.stochastic.input_shape)

        # define the generative convolutional blocks with the skip connections
        # here we assume the skip connections to be of the same shape as `tensor_shp` : this does not work with
        # with every configuration of the generative model. Making the arhitecture more general requires to have
        # a top-down __init__() method such as to take the shapes of the above generative block skip connections as input.
        p_skips = None if (top or no_skip) else [tensor_shp] * len(convolutions)
        self.p_convs = DeterministicBlocks(self.p_proj.output_shape, convolutions[::-1],
                                           aux_shape=p_skips, transposed=True,
                                           in_residual=False, Block=Block, dropout=p_dropout, **kwargs)

        self._p_output_shape = {'d': self.p_convs.output_shape, 'aux': self.p_convs.hidden_shapes}

    @property
    def q_output_shape(self) -> Dict[str, Tuple[int]]:
        """size of the output tensors for the inference path"""
        return self._q_output_shape

    @property
    def p_output_shape(self) -> Tuple[int]:
        """size of the output tensor for the generative path"""
        return self._p_output_shape

    def infer(self, data: Dict[str, Tensor], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform a forward pass through the inference layers and sample the posterior.

        :param data: input data
        :param kwargs: additional parameters passed to the stochastic layer
        :return: (output data, variational data)
        """
        x = data.get('x')
        aux = data.get('aux', None)
        if self._no_skip:
            aux = None

        aux = [aux for _ in range(len(self.q_convs))] if aux is not None else None
        x, _ = self.q_convs(x, aux)

        z, q_data = self.stochastic(x, inference=True, **kwargs)
        z = self.q_proj(z)

        return {'x': z, 'aux': x}, q_data

    def forward(self, data: dict, posterior: Optional[dict], **kwargs) -> Tuple[
        Dict, Dict[str, List]]:
        """
        Perform a forward pass through the generative model and compute KL if posterior data is available

        :param data: data from the above stage forward pass
        :param posterior: dictionary representing the posterior from same stage inference pass
        :return: (dict('d' : d, 'aux : [aux]), dict('kl': [kl], **auxiliary) )
        """
        d = data.get('d', None)
        aux = data.get('aux', None)
        if self._no_skip:
            aux = None

        # sample p(z | d)
        z_p, p_data = self.stochastic(d, inference=False, sample=posterior is None, **kwargs)

        # compute KL(q | p)
        if posterior is not None:
            loss_data = self.stochastic.loss(posterior, p_data, **kwargs)
            z = posterior.get('z')
        else:
            loss_data = {}
            z = z_p

        # project z
        z = self.p_proj(z)

        # pass through convolutions
        d, skips = self.p_convs(z, aux=aux)

        output_data = {'d': d, 'aux': skips}
        return output_data, loss_data


class LvaeStage(VaeStage):
    def __init__(self,
                 input_shape: Dict[str, Tuple[int]],
                 convolutions: List[Tuple[int]],
                 stochastic: Tuple,
                 top: bool = False,
                 bottom: bool = False,
                 q_dropout: float = 0,
                 p_dropout: float = 0,
                 Block: Any = GatedResNet,
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
        :param top: is top layer
        :param bottom: is bottom layer
        :param q_dropout: inference dropout value
        :param p_dropout: generative dropout value
        :param kwargs: others arguments passed to the block constructors (both convolutions and stochastic)
        """
        super().__init__(input_shape, convolutions, stochastic, top=top, bottom=bottom, p_dropout=p_dropout,
                         q_dropout=q_dropout, Block=Block, **kwargs)
        self.q_proj = None
        # get the tensor shape of the output of the deterministic path
        top_shape = self._q_output_shape.get('aux')
        # modify the output of the inference path to be only deterministic
        self._q_output_shape['x'] = top_shape

        topdown = top_shape if not top else None
        conv = convolutions[-1]
        if isinstance(conv, list):
            conv = [conv[0], conv[1], 1, conv[-1]]
        self.merge = Block(top_shape, conv, aux_shape=topdown, transposed=False, in_residual=True, dropout=p_dropout,
                           **kwargs)

    def infer(self, data: Dict[str, Tensor], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform a forward pass through the inference layers and sample the posterior.

        :param data: input data
        :param kwargs: additional parameters passed to the stochastic layer
        :return: (output data, variational data)
        """
        x = data.get('x')
        aux = data.get('aux', None)
        if self._no_skip:
            aux = None

        aux = [aux for _ in range(len(self.q_convs))] if aux is not None else None
        x, _ = self.q_convs(x, aux)

        return {'x': x, 'aux': x}, {'h': x}

    def forward(self, data: dict, posterior: Optional[dict], debugging: bool = False, **kwargs) -> Tuple[
        Dict, Dict[str, List[Tensor]]]:
        """
        Perform a forward pass through the generative model and compute KL if posterior data is available

        :param data: data from the above stage forward pass
        :param posterior: dictionary representing the posterior
        :return: (dict('d' : d, 'aux : [aux]), dict('kl': [kl], **auxiliary) )
        """
        d = data.get('d', None)

        # sample p(z | d)
        z_p, p_data = self.stochastic(d, inference=False, sample=posterior is None, **kwargs)

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

        # project z
        z = self.p_proj(z)

        # pass through convolutions
        aux = data.get('aux', None)
        if self._no_skip:
            aux = None

        d, skips = self.p_convs(z, aux)

        output_data = {'d': d, 'aux': skips}
        return output_data, loss_data


class BivaIntermediateStage(BaseStage):
    def __init__(self,
                 input_shape: Dict[str, Tuple[int]],
                 convolutions: List[Tuple[int]],
                 stochastic: Union[Dict, Tuple[Dict]],
                 top: bool = False,
                 bottom: bool = False,
                 q_dropout: float = 0,
                 p_dropout: float = 0,
                 no_skip: bool = False,
                 conditional_bu: bool = False,
                 Block: Any = GatedResNet,
                 merge_kernel: int = 3,
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
        :param bottom: is bottom layer
        :param top: is top layer
        :param q_dropout: inference dropout value
        :param p_dropout: generative dropout value
        :param no_skip: do not use skip connections
        :param conditional_bu: condition BU prior on p(z_TD)
        :param aux_shape: auxiliary input tensor shape as a tuple of integers (B, H, *D)
        :param kwargs: others arguments passed to the block constructors (both convolutions and stochastic)
        """
        super().__init__(input_shape, convolutions, stochastic, top=top, bottom=bottom, q_dropout=q_dropout,
                         p_dropout=p_dropout, Block=Block, no_skip=no_skip)

        self._conditional_bu = conditional_bu
        self._merge_kernel = merge_kernel

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

        # mute skip connections
        if no_skip:
            aux_shape = None

        # define inference convolutional blocks
        in_residual = not bottom
        q_bu_aux = [aux_shape for _ in convolutions] if aux_shape is not None else None
        self.q_bu_convs = DeterministicBlocks(bu_shp, convolutions, aux_shape=q_bu_aux, transposed=False,
                                              in_residual=in_residual, dropout=q_dropout, Block=Block, **kwargs)

        q_td_aux = [self.q_bu_convs.output_shape for _ in convolutions]
        self.q_td_convs = DeterministicBlocks(td_shp, convolutions, aux_shape=q_td_aux, transposed=False,
                                              in_residual=in_residual, dropout=q_dropout, Block=Block, **kwargs)

        # shape of the output of the inference path and input tensor from the generative path
        top_tensor_shp = self.q_td_convs.output_shape
        aux_shape = shp_cat([top_tensor_shp, top_tensor_shp], 1)

        # define the BU stochastic layer
        bu_top = False if conditional_bu else top
        self.bu_stochastic = StochasticBlock(bu_stochastic, top_tensor_shp, top=bu_top, **kwargs)
        self.bu_proj = AsFeatureMap(self.bu_stochastic.output_shape, self.bu_stochastic.input_shape, **kwargs)

        # define the TD stochastic layer
        self.td_stochastic = StochasticBlock(td_stochastic, top_tensor_shp, top=top, **kwargs)

        self._q_output_shape = {'x_bu': self.bu_proj.output_shape,
                                'x_td': top_tensor_shp,
                                'aux': aux_shape}

        ### GENERATIVE MODEL

        # TD merge layer
        h_shape = self._q_output_shape.get('x_td', None) if not self._top else None
        conv = self._convolutions[::-1][-1]
        if isinstance(conv, list) or isinstance(conv, tuple):
            conv = [conv[0], merge_kernel, 1,
                    conv[-1]]  # in the original implementation, this depends on the parameters of the above layers
        self.merge = Block(h_shape, conv, aux_shape=h_shape, transposed=False, in_residual=True,
                           dropout=p_dropout,
                           **kwargs)

        # alternative: define the condition p(z_bu | z_td, ...)
        if conditional_bu:
            self.bu_condition = Block(self.bu_stochastic.output_shape, conv, aux_shape=h_shape, transposed=False,
                                      in_residual=False,
                                      dropout=p_dropout, **kwargs)
        else:
            self.bu_condition = None

        # merge latent variables
        z_shp = shp_cat([self.bu_stochastic.output_shape, self.td_stochastic.output_shape], 1)
        self.z_proj = AsFeatureMap(z_shp, self.bu_stochastic.input_shape)

        # define the generative convolutional blocks with the skip connections
        # here we assume the skip connections to be of the same shape as `top_tensor_shape` : this does not work with
        # with every configuration of the generative model. Making the arhitecture more general requires to have
        # a top-down __init__() method such as to take the shapes of the above generative block skip connections as input.
        p_skips = None if (top or no_skip) else [top_tensor_shp] * len(convolutions)
        self.p_convs = DeterministicBlocks(self.z_proj.output_shape, self._convolutions[::-1],
                                           aux_shape=p_skips, transposed=True,
                                           in_residual=False, Block=Block, dropout=p_dropout, **kwargs)

        self._p_output_shape = {'d': self.p_convs.output_shape, 'aux': self.p_convs.hidden_shapes}

    @property
    def q_output_shape(self) -> Dict[str, Tuple[int]]:
        """size of the output tensors for the inference path"""
        return self._q_output_shape

    @property
    def p_output_shape(self) -> Tuple[int]:
        """size of the output tensor for the generative path"""
        return self._p_output_shape

    def infer(self, data: Dict[str, Tensor], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
        if self._no_skip:
            aux = None

        # BU path
        bu_aux = [aux for _ in range(len(self.q_bu_convs))] if aux is not None else None
        x_bu, _ = self.q_bu_convs(x_bu, aux=bu_aux)
        # z_bu ~ q(x)
        z_bu, bu_q_data = self.bu_stochastic(x_bu, inference=True, **kwargs)
        z_bu_proj = self.bu_proj(z_bu)

        # TD path
        td_aux = [x_bu for _ in range(len(self.q_td_convs))]
        x_td, _ = self.q_td_convs(x_td, aux=td_aux)
        td_q_data = {'z': z_bu, 'h': x_td}  # note h = d_q(x)

        # skip connection
        aux = torch.cat([x_bu, x_td], 1)

        return {'x_bu': z_bu_proj, 'x_td': x_td, 'aux': aux}, {'z_bu': z_bu, 'bu': bu_q_data, 'td': td_q_data}

    def forward(self, data: Dict, posterior: Optional[dict], debugging: bool = False, **kwargs) -> Tuple[
        Dict, Dict[str, List[Tensor]]]:
        """
        Perform a forward pass through the generative model and compute KL if posterior data is available

        :param d: previous hidden state
        :param posterior: dictionary representing the posterior
        :return: (hidden state, dict('kl': [kl], **auxiliary))
        """

        d = data.get('d', None)

        if posterior is not None:
            # sample posterior and compute KL using prior
            bu_q_data = posterior.get('bu')
            td_q_data = posterior.get('td')
            z_bu_q = posterior.get('z_bu')

            # top-down: compute the posterior using the bottom-up hidden state and top-down hidden state
            # p(z_td | d_top)
            _, td_p_data = self.td_stochastic(d, inference=False, sample=False, **kwargs)

            # merge d_top with h = d_q(x)
            h = td_q_data.get('h')
            h = self.merge(h, aux=d)

            # z_td ~ q(z_td | h_bu_td)
            z_td_q, td_q_data = self.td_stochastic(h, inference=True, sample=True, **kwargs)

            # compute log q_bu(z_i | x) - log p_bu(z_i) (+ additional data)
            td_loss_data = self.td_stochastic.loss(td_q_data, td_p_data, **kwargs)

            # conditional BU prior
            if self.bu_condition is not None:
                d_ = self.bu_condition(z_td_q, aux=d)
            else:
                d_ = d

            # bottom-up: retrieve data from the inference path
            # z_bu ~ p(d_top)
            _, bu_p_data = self.bu_stochastic(d_, inference=False, sample=False, **kwargs)

            # compute log q_td(z_i | x, z_{>i}) - log p_td(z_i) (+ additional data)
            bu_loss_data = self.bu_stochastic.loss(bu_q_data, bu_p_data, **kwargs)

            # merge samples
            z = torch.cat([z_td_q, z_bu_q], 1)

        else:
            # sample priors
            # top-down
            z_td_p, td_p_data = self.td_stochastic(d, inference=False, sample=True, **kwargs)  # prior

            # conditional BU prior
            if self.bu_condition is not None:
                d_ = self.bu_condition(z_td_p, aux=d)
            else:
                d_ = d

            # bottom-up
            z_bu_p, bu_p_data = self.bu_stochastic(d_, inference=False, sample=True, **kwargs)  # prior

            bu_loss_data, td_loss_data = {}, {}

            # merge samples
            z = torch.cat([z_td_p, z_bu_p], 1)

        # projection
        z = self.z_proj(z)

        # pass through convolutions
        aux = data.get('aux', None)
        if self._no_skip:
            aux = None

        d, skips = self.p_convs(z, aux=aux)

        # gather data
        loss_data = DataCollector()
        loss_data.extend(td_loss_data)
        loss_data.extend(bu_loss_data)

        output_data = {'d': d, 'aux': skips}
        return output_data, loss_data


class BivaTopStage_simpler(VaeStage):
    """
    This is the BivaTopStage without the additional BU-TD merge layer.
    """

    def __init__(self, input_shape: Dict[str, Tuple[int]], *args, **kwargs):
        bu_shp = input_shape.get('x_bu')
        td_shp = input_shape.get('x_td')

        tensor_shp = shp_cat([bu_shp, td_shp], 1)
        concat_shape = {'x': tensor_shp}

        super().__init__(concat_shape, *args, **kwargs)

    def infer(self, data: Dict[str, Tensor], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        x_bu = data.pop('x_bu')
        x_td = data.pop('x_td')
        data['x'] = torch.cat([x_bu, x_td], 1)

        return super().infer(data, **kwargs)


class BivaTopStage(BaseStage):
    def __init__(self, input_shape: Dict[str, Tuple[int]],
                 convolutions: List[Tuple[int]],
                 stochastic: Union[Dict, Tuple[Dict]],
                 top: bool = False,
                 bottom: bool = False,
                 q_dropout: float = 0,
                 p_dropout: float = 0,
                 no_skip: bool = False,
                 Block: Any = GatedResNet,
                 **kwargs):
        """
        BIVA: https://arxiv.org/abs/1902.02102

        Define a Bidirectional Variational Autoencoder top stage containing:
        - a sequence of convolutional blocks for the bottom-up inference model (BU)
        - a sequence of convolutional blocks for the top-down inference model (TD)
        - a convolutional block to merge BU and TD
        - a sequence of convolutional blocks for the generative model
        - a stochastic layer (z_L)

        :param input_shape: dictionary describing the input tensor shape (B, H, *D)
        :param convolution: list of tuple describing a convolutional block (filters, kernel_size, stride)
        :param stochastic: dictionary describing the stochastic layer: units or (units, kernel_size, discrete, K)
        :param bottom: is bottom layer
        :param top: is top layer
        :param q_dropout: inference dropout value
        :param p_dropout: generative dropout value
        :param no_skip: do not use skip connections
        :param aux_shape: auxiliary input tensor shape as a tuple of integers (B, H, *D)
        :param kwargs: others arguments passed to the block constructors (both convolutions and stochastic)
        """
        super().__init__(input_shape, convolutions, stochastic, top=top, bottom=bottom, q_dropout=q_dropout,
                         p_dropout=p_dropout, Block=Block, no_skip=no_skip)
        top = True

        if 'x' in input_shape.keys():
            bu_shp = td_shp = input_shape.get('x')
            aux_shape = None
        else:
            bu_shp = input_shape.get('x_bu')
            td_shp = input_shape.get('x_td')
            aux_shape = input_shape.get('aux')

        # mute skip connections
        if no_skip:
            aux_shape = None

        # define inference BU and TD paths
        in_residual = not bottom
        q_bu_aux = [aux_shape for _ in convolutions] if aux_shape is not None else None
        self.q_bu_convs = DeterministicBlocks(bu_shp, convolutions, aux_shape=q_bu_aux, transposed=False,
                                              in_residual=in_residual, dropout=q_dropout, Block=Block, **kwargs)

        q_td_aux = [self.q_bu_convs.output_shape for _ in convolutions]
        self.q_td_convs = DeterministicBlocks(td_shp, convolutions, aux_shape=q_td_aux, transposed=False,
                                              in_residual=in_residual, dropout=q_dropout, Block=Block, **kwargs)

        # merge BU and TD paths
        conv = convolutions[-1]
        self.q_top = Block(shp_cat([self.q_bu_convs.output_shape, self.q_td_convs.output_shape], 1),
                           [conv[0], conv[1], 1, conv[-1]], dropout=q_dropout,
                           residual=True, **kwargs)
        top_tensor_shp = self.q_top.output_shape

        # stochastic layer
        self.stochastic = StochasticBlock(stochastic, top_tensor_shp, top=top, **kwargs)

        self._q_output_shape = {}  # no output shape (top layer)

        ### GENERATIVE MODEL

        # map sample back to a feature map
        self.z_proj = AsFeatureMap(self.stochastic.output_shape, self.stochastic.input_shape)

        # define the generative convolutional blocks with the skip connections
        p_skips = None
        self.p_convs = DeterministicBlocks(self.z_proj.output_shape, self._convolutions[::-1],
                                           aux_shape=p_skips, transposed=True,
                                           in_residual=False, Block=Block, dropout=p_dropout, **kwargs)

        self._p_output_shape = {'d': self.p_convs.output_shape, 'aux': self.p_convs.hidden_shapes}

    def infer(self, data: Dict[str, Tensor], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
        if self._no_skip:
            aux = None

        # BU path
        bu_aux = [aux for _ in range(len(self.q_bu_convs))] if aux is not None else None
        x_bu, _ = self.q_bu_convs(x_bu, aux=bu_aux)

        # TD path
        td_aux = [x_bu for _ in range(len(self.q_td_convs))]
        x_td, _ = self.q_td_convs(x_td, aux=td_aux)

        # merge BU and TD
        x = torch.cat([x_bu, x_td], 1)
        x = self.q_top(x)

        # sample top layer
        z, q_data = self.stochastic(x, inference=True, **kwargs)

        return {}, q_data

    def forward(self, data: Dict, posterior: Optional[dict], **kwargs) -> Tuple[
        Dict, Dict[str, List]]:
        """
        Perform a forward pass through the generative model and compute KL if posterior data is available

        :param data: data from the above stage forward pass
        :param posterior: dictionary representing the posterior
        :return: (hidden state, dict('kl': [kl], **auxiliary) )
        """
        d = data.get('d', None)

        if posterior is not None:
            # get p(z | d)
            _, p_data = self.stochastic(d, inference=False, sample=False, **kwargs)

            # compute KL(q | p)
            loss_data = self.stochastic.loss(posterior, p_data, **kwargs)
            z = posterior.get('z')
        else:
            loss_data = {}
            z, p_data = self.stochastic(d, inference=False, sample=True, **kwargs)

        # project z
        z = self.z_proj(z)

        # pass through convolutions
        aux = data.get('aux', None)
        if self._no_skip:
            aux = None

        d, skips = self.p_convs(z, aux=aux)

        output_data = {'d': d, 'aux': skips}
        return output_data, loss_data

    @property
    def q_output_shape(self) -> Dict[str, Tuple[int]]:
        """size of the output tensors for the inference path"""
        return self._q_output_shape

    @property
    def p_output_shape(self) -> Tuple[int]:
        """size of the output tensor for the generative path"""
        return self._p_output_shape


def BivaStage(input_shape: Dict[str, Tuple[int]],
              convolutions: List[Tuple[int]],
              stochastic: Union[Dict, Tuple[Dict]],
              top: bool = False,
              **kwargs):
    """
    BIVA: https://arxiv.org/abs/1902.02102

    Define a Bidirectional Variational Autoencoder stage containing:
    - a sequence of convolutional blocks for the bottom-up inference model (BU)
    - a sequence of convolutional blocks for the top-down inference model (TD)
    - a sequence of convolutional blocks for the generative model
    - two stochastic layers (BU and TD)

    This is not an op-for-op implementation of the original Tensorflow version.

    :param input_shape: dictionary describing the input tensor shape (B, H, *D)
    :param convolution: list of tuple describing a convolutional block (filters, kernel_size, stride)
    :param stochastic: dictionary describing the stochastic layer: units or (units, kernel_size, discrete, K)
    :param top: is top layer
    :param bottom: is bottom layer
    :param q_dropout: inference dropout value
    :param p_dropout: generative dropout value
    :param conditional_bu: condition BU prior on p(z_TD)
    :param aux_shape: auxiliary input tensor shape as a tuple of integers (B, H, *D)
    :param kwargs: others arguments passed to the block constructors (both convolutions and stochastic)
    """

    if top:
        return BivaTopStage(input_shape, convolutions, stochastic, top=top, **kwargs)
    else:
        return BivaIntermediateStage(input_shape, convolutions, stochastic, top=top, **kwargs)
