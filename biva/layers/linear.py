import numpy as np
import torch
from torch import nn


class NormedLinear(nn.Module):
    """
    Linear layer with normalization
    """

    def __init__(self, in_features, out_features, dim=-1, weightnorm=True):
        super(NormedLinear, self).__init__()
        """
        args:
            in_features (in): number of input features
            out_features (int): number of output features
            dim (int): dimension to aply transformation to
            weightnorm (bool): use weight normalization
        """
        self.initialized = False
        self._in_features = in_features
        self._out_features = out_features
        self.dim = dim
        self.linear = nn.Linear(self._in_features, out_features)
        # add batch norm
        if not weightnorm:
            self.weightnorm = False
        else:
            self.weightnorm = True
            self.linear = nn.utils.weight_norm(self.linear, dim=0, name="weight")

    def forward(self, x):
        # reshape in
        shp = list(x.size())
        dim = self.dim if self.dim >= 0 else x.dim() + self.dim
        if dim < x.dim() - 1:
            x = x.view(*x.size()[:dim + 1], -1)
            x = x.transpose(-1, -2).contiguous()
            shp_2 = list(x.shape)
            x = x.view(-1, x.size(-1))
            permute = True
        else:
            x = x.view(-1, x.size(-1))
            permute = False
        # init and transform
        if not self.initialized:
            self.init_parameters(x)
        x = self.linear(x)
        # reshape out
        shp[dim] = self._out_features
        if permute:
            shp_2[-1] = self._out_features
            x = x.view(shp_2).transpose(-1, -2)
            x = x.view(shp)
        else:
            x = x.view(shp)
        return x

    @property
    def input_shape(self):
        return (-1, self._in_features)

    @property
    def output_shape(self):
        return (-1, self._out_features)

    def init_parameters(self, x, init_scale=0.05, eps=1e-8):
        if self.weightnorm:
            # initial values
            self.linear._parameters['weight_v'].data.normal_(mean=0, std=init_scale)
            self.linear._parameters['weight_g'].data.fill_(1.)
            self.linear._parameters['bias'].data.fill_(0.)
            init_scale = .01
            # data dependent init
            x = self.linear(x)
            m_init, v_init = torch.mean(x, 0), torch.var(x, 0)
            scale_init = init_scale / torch.sqrt(v_init + eps)
            self.linear._parameters['weight_g'].data = self.linear._parameters['weight_g'].data * scale_init.view(
                self.linear._parameters['weight_g'].data.size())
            self.linear._parameters['bias'].data = self.linear._parameters['bias'].data - m_init * scale_init
            self.initialized = True
            return scale_init[None, :] * (x - m_init[None, :])


class NormedDense(nn.Module):
    """
    Dense layer with normalization
    """

    def __init__(self, tensor_shape, out_features, weightnorm=True):
        super(NormedDense, self).__init__()
        """
        args:
            tensor_shape (tuple): input tensor shape (B x C x D)
            out_features (int): number of output features
            weight (bool): use weight normalization
        """
        self.initialized = False
        self._input_shp = tensor_shape
        self.input_features = int(np.prod(tensor_shape[1:]))
        self._output_shp = (-1, out_features)
        self.linear = nn.Linear(self.input_features, out_features)
        # add batch norm
        if not weightnorm:
            self.weightnorm = False
        else:
            self.weightnorm = True
            self.linear = nn.utils.weight_norm(self.linear, dim=0, name="weight")

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        if not self.initialized:
            self.init_parameters(x)
        x = self.linear(x)
        return x

    @property
    def input_shape(self):
        return self._input_shp

    @property
    def output_shape(self):
        return self._output_shp

    def init_parameters(self, x, init_scale=0.05, eps=1e-8):
        if self.weightnorm:
            # initial values
            self.linear._parameters['weight_v'].data.normal_(mean=0, std=init_scale)
            self.linear._parameters['weight_g'].data.fill_(1.)
            self.linear._parameters['bias'].data.fill_(0.)
            init_scale = .01
            # data dependent init
            x = self.linear(x)
            m_init, v_init = torch.mean(x, 0), torch.var(x, 0)
            scale_init = init_scale / torch.sqrt(v_init + eps)
            self.linear._parameters['weight_g'].data = self.linear._parameters['weight_g'].data * scale_init.view(
                self.linear._parameters['weight_g'].data.size())
            self.linear._parameters['bias'].data = self.linear._parameters['bias'].data - m_init * scale_init
            self.initialized = True
            return scale_init[None, :] * (x - m_init[None, :])
