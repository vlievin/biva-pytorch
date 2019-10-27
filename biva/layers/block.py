from typing import *

from .convolution import *
from .linear import *


class GatedResNet(nn.Module):
    def __init__(self, input_shape: Tuple, dim: Tuple, aux_shape: Optional[Tuple] = None, weightnorm: bool = True,
                 act: nn.Module = nn.ReLU,
                 transposed: bool = False, dropout: Optional[float] = None, residual: bool = True, **kwargs: Any):
        """
        args:
            input_shape (tuple): input tensor shape (B x C x *D)
            dim (tuple): convolution dimensions (filters, kernel_size, stride)
            aux_shape (tuple): auxiliary input tensor shape (B x C x *D). None means no auxialiary input
            weightnorm (bool): use weight normalization
            act (nn.Module): activation function class
            transposed (bool): transposed or not
            dropout (float): dropout value. None is no dropout
            residual (bool): use residual connections
        """
        super(GatedResNet, self).__init__()

        # define convolution and transposed convolution objects
        conv_obj = nn.Conv1d if len(input_shape[2:]) == 1 else nn.Conv2d
        deconv_obj = nn.ConvTranspose1d if len(input_shape[2:]) == 1 else nn.ConvTranspose2d

        # some parameters
        C_in = input_shape[1]
        self.residual = residual
        self.transposed = transposed
        self.act = act()

        # conv 1
        conv1 = conv_obj(C_in, dim[0], dim[1], 1)
        self.conv1 = PaddedNormedConv(input_shape, conv1, weightnorm=weightnorm)
        shp = self.conv1.output_shape

        # dropout
        self.dropout = nn.Dropout(dropout) if dropout is not None else dropout

        # conv 2
        if self.transposed and dim[2] > 1:
            conv2 = deconv_obj(dim[0], 2 * dim[0], dim[1], dim[2])
        else:
            conv2 = conv_obj(dim[0], 2 * dim[0], dim[1], dim[2])

        self.conv2 = PaddedNormedConv(shp, conv2, weightnorm=weightnorm)

        # input / output shapes
        shp = list(self.conv2.output_shape)
        shp[1] = shp[1] // 2  # gated
        self._input_shape = input_shape
        self._output_shape = tuple(shp)
        self.aux_shape = aux_shape

        # residual connections
        self.residual_op = ResidualConnection(self._input_shape, shp, residual)

        # aux op
        if aux_shape is not None:
            if list(aux_shape[2:]) > list(input_shape[2:]):
                stride = tuple(np.asarray(aux_shape[2:]) // np.asarray(input_shape[2:]))
                aux_conv = conv_obj(aux_shape[1], dim[0], dim[1], stride)
                self.aux_op = PaddedNormedConv(aux_shape, aux_conv, weightnorm=weightnorm)

            elif list(aux_shape[2:]) < list(input_shape[2:]):
                stride = tuple(np.asarray(input_shape[2:]) // np.asarray(aux_shape[2:]))
                aux_conv = deconv_obj(aux_shape[1], dim[0], dim[1], stride)
                self.aux_op = PaddedNormedConv(aux_shape, aux_conv, weightnorm=weightnorm)

            else:
                aux_conv = conv_obj(aux_shape[1], dim[0], 1, 1)  # conv with kernel 1
                self.aux_op = PaddedNormedConv(aux_shape, aux_conv, weightnorm=weightnorm)

        else:
            self.aux_op = None

    def forward(self, x: torch.Tensor, aux: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:

        # input activation: x = act(x)
        x_act = self.act(x) if self.residual else x

        # conv 1: y = conv(x)
        y = self.conv1(x_act)

        # merge aux with x: y = y + f(aux)
        y = y + self.aux_op(self.act(aux)) if self.aux_op is not None else y

        # y = act(y)
        y = self.act(y)

        # dropout
        y = self.dropout(y) if self.dropout else y

        # conv 2: y = conv(y)
        y = self.conv2(y)

        # gate: y = y_1 * sigmoid(y_2)
        h_stack1, h_stack2 = y.chunk(2, 1)
        sigmoid_out = torch.sigmoid(h_stack2)
        y = (h_stack1 * sigmoid_out)

        # resiudal connection: y = y + x
        y = self.residual_op(y, x)

        return y

    @property
    def input_shape(self) -> Tuple:
        return self._input_shape

    @property
    def output_shape(self) -> Tuple:
        return self._output_shape


class ResMLP(nn.Module):
    def __init__(self, input_shape: Tuple[int], dim: int, aux_shape: Optional[int] = None, weightnorm: bool = True,
                 act: nn.Module = nn.ReLU, transposed: bool = False, dropout: float = None, residual: bool = True,
                 mlp_layers: int = 1,
                 **kwargs: Any):
        super().__init__()

        # convert parameters
        ninp = input_shape[1]
        nhid = dim
        naux = aux_shape[1] if aux_shape is not None else 0
        nlayers = mlp_layers

        # params
        self._input_shape = input_shape
        self._output_shape = (-1, dim)

        # model
        self.act = act()
        self.ninp = ninp
        self.naux = naux
        if naux is None:
            naux = 0
        self.residual = NormedLinear(ninp, nhid, weightnorm) if ninp != nhid else None
        if residual:
            layers = [act(), NormedLinear(ninp + naux, nhid, weightnorm), nn.BatchNorm1d(nhid)]
        else:
            layers = [NormedLinear(ninp + naux, nhid, weightnorm), nn.BatchNorm1d(nhid)]
        layers += (nlayers - 1) * [act(), NormedLinear(nhid, nhid, weightnorm), nn.BatchNorm1d(nhid)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, aux=None):
        if self.residual is not None:
            r = self.residual(x)
        else:
            r = x
        if aux is not None:
            x = torch.cat([x, aux], 1)
        return r + self.layers(x)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape
