import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def getSAMEPadding(tensor_shape, conv):
    """
    return the padding to apply to a given convolution such as it reproduces the 'SAME' behavior from Tensorflow
    This works as well for pooling layers.
    args:
        tensor_shape (tuple): input tensor shape (B x C x D)
        conv (nn.Module): convolution object
    returns:
        sym_padding, unsym_padding: symmetric and unsymmetric padding to apply,

    """
    tensor_shape = np.asarray(tensor_shape)[2:]
    kernel_size = np.asarray(conv.kernel_size)
    pad = np.asarray(conv.padding)
    dilation = np.asarray(conv.dilation) if hasattr(conv, 'dilation') else 1
    stride = np.asarray(conv.stride)
    # handle pooling layers
    if not hasattr(conv, 'transposed'):
        conv.transposed = False
    else:
        assert len(tensor_shape) == len(kernel_size), "tensor is not the same dimension as the kernel"
    if not conv.transposed:
        effective_filter_size = (kernel_size - 1) * dilation + 1
        output_size = (tensor_shape + stride - 1) // stride
        padding_input = np.maximum(0, (output_size - 1) * stride + (kernel_size - 1) * dilation + 1 - tensor_shape)
        odd_padding = (padding_input % 2 != 0)
        sym_padding = tuple(padding_input // 2)
        unsym_padding = [y for x in odd_padding for y in [0, int(x)]]
    else:
        padding_input = kernel_size - stride
        sym_padding = None
        unsym_padding = [y for x in padding_input for y in
                         [-int(np.floor(int(x) / 2)), -int(np.floor(int(x) / 2) + int(x) % 2)]]
    return sym_padding, unsym_padding


class PaddedConv(nn.Module):
    """
    wrapps convolution instance with SAME padding (as in tensorflow)
    This requires providing the input tensor shape.
    """

    def __init__(self, tensor_shape, conv):
        """
        args:
            tensor_shape (tuple): input tensor shape
            conv (nn.Module): convolution instance
        """
        super(PaddedConv, self).__init__()
        # input and output shapes
        self._input_shp = tensor_shape
        self._output_shp = np.asarray(tensor_shape)
        stride_factor = np.asarray(conv.stride)
        self._output_shp[2:] = self._output_shp[2:] // stride_factor if (
                not hasattr(conv, 'transposed') or not conv.transposed) else self._output_shp[2:] * stride_factor
        self._output_shp[1] = conv.out_channels if hasattr(conv, 'out_channels') else self._output_shp[1]
        self._output_shp = tuple(self._output_shp.astype(int))
        # get paddings
        sym_padding, self.unsym_padding = getSAMEPadding(tensor_shape, conv)
        if not conv.transposed:
            conv.padding = sym_padding
        self.conv = conv

    def forward(self, x):
        x = F.pad(x, self.unsym_padding) if not self.conv.transposed else x
        x = self.conv(x)
        x = F.pad(x, self.unsym_padding) if self.conv.transposed else x
        return x

    def init_parameters(self, x):
        x = F.pad(x, self.unsym_padding) if not self.conv.transposed else x
        self.conv.init_parameters(x)

    @property
    def input_shape(self):
        return self._input_shp

    @property
    def output_shape(self):
        return self._output_shp


class PaddedNormedConv(nn.Module):
    """
    A class to handle both normalization and SAME padding (as in tensorflow) for convolutions.
    This class also handles data dependent initialization for weight normalization
    """

    def __init__(self, tensor_shape, conv, weightnorm=True):
        """
        args:
            tensor_shape (tuple): input tensor shape (B x C x D)
            conv (nn.Module): convolution instance of type Conv1d, ConvTranspose1d, Conv2d or ConvTranspose2d
            weightnorm (bool): use weight normalization
        """
        super(PaddedNormedConv, self).__init__()
        self.initialized = False

        # paddding
        self._input_shp = tensor_shape
        self._output_shp = np.asarray(tensor_shape)
        stride_factor = np.asarray(conv.stride)
        self._output_shp[2:] = self._output_shp[2:] // stride_factor if (
                not hasattr(conv, 'transposed') or not conv.transposed) else self._output_shp[2:] * stride_factor
        self._output_shp[1] = conv.out_channels if hasattr(conv, 'out_channels') else self._output_shp[1]
        self._output_shp = tuple(self._output_shp.astype(int))

        # get paddings
        sym_padding, self.unsym_padding = getSAMEPadding(tensor_shape, conv)
        if not conv.transposed:
            conv.padding = sym_padding
        self.conv = conv

        # add batch norm
        if not weightnorm:
            self.weightnorm = False
        else:
            self.weightnorm = True
            dim = 1 if self.conv.transposed else 0
            self.conv = nn.utils.weight_norm(self.conv, dim=dim, name="weight")

    def forward(self, x):
        x = F.pad(x, self.unsym_padding) if not self.conv.transposed else x
        if not self.initialized:
            self.init_parameters(x)
        x = self.conv(x)

        x = F.pad(x, self.unsym_padding) if self.conv.transposed else x
        return x

    @property
    def input_shape(self):
        return self._input_shp

    @property
    def output_shape(self):
        return self._output_shp

    def init_parameters(self, x, init_scale=0.05, eps=1e-8):
        self.initialized = True
        if self.weightnorm:
            # initial values
            self.conv._parameters['weight_v'].data.normal_(mean=0, std=init_scale)
            self.conv._parameters['weight_g'].data.fill_(1.)
            self.conv._parameters['bias'].data.fill_(0.)
            init_scale = .01
            # data dependent init
            x = self.conv(x)
            t = x.view(x.size()[0], x.size()[1], -1)
            t = t.permute(0, 2, 1).contiguous()
            t = t.view(-1, t.size()[-1])
            m_init, v_init = torch.mean(t, 0), torch.var(t, 0)
            scale_init = init_scale / torch.sqrt(v_init + eps)
            if self.conv.transposed:
                self.conv._parameters['weight_g'].data = self.conv._parameters['weight_g'].data * scale_init[None,
                                                                                                  :].view(
                    self.conv._parameters['weight_g'].data.size())
                self.conv._parameters['bias'].data = self.conv._parameters['bias'].data - m_init * scale_init
            else:
                self.conv._parameters['weight_g'].data = self.conv._parameters['weight_g'].data * scale_init[:,
                                                                                                  None].view(
                    self.conv._parameters['weight_g'].data.size())
                self.conv._parameters['bias'].data = self.conv._parameters['bias'].data - m_init * scale_init
            return scale_init[None, :, None, None] * (x - m_init[None, :, None, None]) if len(
                self._input_shp) > 3 else scale_init[None, :, None] * (x - m_init[None, :, None])


class ResidualConnection(nn.Module):
    """
    Handles residual connections for tensors with different shapes.
    Apply padding and/or avg pooling to the input when necessary
    """

    def __init__(self, input_shape, output_shape, residual=True):
        """
        args:
            input_shape (tuple): input module shape x
            output_shape (tuple): output module shape y=f(x)
            residual (bool): apply residual conenction y' = y+x = f(x)+x
        """
        super().__init__()
        self.residual = residual
        self.input_shape = input_shape
        self.output_shape = output_shape
        is_text = len(input_shape) == 3

        # residual: features
        if residual and self.output_shape[1] < self.input_shape[1]:
            pad = int(self.output_shape[1]) - int(self.input_shape[1])
            self.redidual_padding = [0, 0, 0, pad] if is_text else [0, 0, 0, 0, 0, pad]

        elif residual and self.output_shape[1] > self.input_shape[1]:
            pad = int(self.output_shape[1]) - int(self.input_shape[1])
            self.redidual_padding = [0, 0, 0, pad] if is_text else [0, 0, 0, 0, 0, pad]
            warnings.warn("The input has more feature maps than the output. There will be no residual connection for this layer.")
            self.residual = False
        else:
            self.redidual_padding = None

        # residual: dimension
        if residual and list(output_shape)[2:] < list(input_shape)[2:]:
            pool_obj = nn.AvgPool1d if len(output_shape[2:]) == 1 else nn.AvgPool2d
            stride = tuple((np.asarray(input_shape)[2:] // np.asarray(output_shape)[2:]).tolist())
            self.residual_op = PaddedConv(input_shape, pool_obj(3, stride=stride))

        elif residual and list(output_shape)[2:] > list(input_shape)[2:]:
            warnings.warn(
                "The height and width of the output are larger than the input. There will be no residual connection for this layer.")
            # self.residual_op = nn.UpsamplingBilinear2d(size=self.output_shape[2:])
            self.residual = False
        else:
            self.residual_op = None

    def forward(self, y, x):
        if self.residual:
            x = F.pad(x, self.redidual_padding) if self.redidual_padding is not None else x
            x = self.residual_op(x) if self.residual_op is not None else x
            y = y + x
        return y


def getConvolutionOutputShape(tensor_shape, conv):
    """
    compute the output shape of a convolution given the input tensor shape
    args:
        tensor_shape (tuple): input tensor shape (B x C x D)
        conv (nn.Module): convolution object
    returns:
        output_shape (tuple): expected output shape
    """
    assert tensor_shape[1] == conv.in_channels, "tensor and kernel do not have the same nuæber of features"
    tensor_shape = np.asarray(tensor_shape)
    kernel_size = np.asarray(conv.kernel_size)
    pad = np.asarray(conv.padding)
    dilation = np.asarray(conv.dilation)
    stride = np.asarray(conv.stride)
    assert len(tensor_shape) - 2 == len(kernel_size), "tensor is not the same dimension as the kernel"
    dims = tensor_shape[2:]
    if not conv.transposed:
        out_dims = ((dims + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
    else:
        out_padding = np.asarray(conv.output_padding)
        out_dims = dilation * ((dims - 1) * stride - 2 * padding + kernel_size) + out_padding
    out_dims = np.floor(out_dims).astype(int)
    return (tensor_shape[0], conv.out_channels, *out_dims)


if __name__ == '__main__':
    # test getConvolutionOutputShape
    for x in [torch.zeros(1, 3, 33, 33), torch.zeros(1, 3, 64, 64)]:
        for kernel_size in [3, 5, 6, 8]:
            for padding in [1, 2]:
                for stride in [1, 2, 3]:
                    for dilation in [1]:
                        for Conv in [nn.Conv2d, nn.ConvTranspose2d]:
                            conv = Conv(3, 18, kernel_size=kernel_size, padding=padding, stride=stride,
                                        dilation=dilation)
                            x_shp = tuple(x.size())
                            c_shp = getConvolutionOutputShape(x_shp, conv)
                            true_shp = tuple(conv(x).size())
                            assert c_shp[2:] == true_shp[2:]

    # test getPadding
    for x in [torch.zeros(1, 3, 33, 32), torch.zeros(1, 3, 15, 15)]:
        for Conv in [nn.ConvTranspose2d, nn.Conv2d]:
            for stride in [2, 1]:
                for dilation in [1, 2]:
                    for kernel_size in [3, 5, 6]:
                        conv = Conv(3, 5, kernel_size=kernel_size, stride=stride, dilation=dilation)
                        # print("kernel_size:",kernel_size, " stride:", stride," trans.:",conv.transposed, " x:", x.size())
                        x_shp = tuple(x.size())
                        padding, odd_padding = getSAMEPadding(x_shp, conv)
                        if not conv.transposed:
                            conv.padding = padding
                        x_shp = x_shp[2:]
                        expected_shp = tuple([t * stride for t in x_shp]) if conv.transposed else tuple(
                            [t // stride for t in x_shp])
                        y = F.pad(x, odd_padding) if not conv.transposed else x
                        y = conv(y)
                        y = F.pad(y, odd_padding) if conv.transposed else y
                        true_shp = tuple(y.size())[2:]
                        # print(true_shp,expected_shp,np.asarray(true_shp)-np.asarray(expected_shp),kernel_size-stride,'\n')
                        assert true_shp >= expected_shp

    # test SAMEpaddingConv
    for x in [torch.zeros(1, 3, 33, 32), torch.zeros(1, 3, 15, 15)]:
        for Conv in [nn.ConvTranspose2d, nn.Conv2d]:
            for stride in [2, 1]:
                for dilation in [1, 2]:
                    for kernel_size in [3, 5, 6]:
                        conv = Conv(3, 5, kernel_size=kernel_size, stride=stride, dilation=dilation)
                        # print("kernel_size:",kernel_size, " stride:", stride," trans.:",conv.transposed, " x:", x.size())
                        x_shp = tuple(x.size())
                        conv = SAMEpaddingConv(x_shp, conv)
                        x_shp = x_shp[2:]
                        expected_shp = tuple([t * stride for t in x_shp]) if conv.conv.transposed else tuple(
                            [t // stride for t in x_shp])
                        y = conv(x)
                        true_shp = tuple(y.size())[2:]
                        # print(true_shp,expected_shp,np.asarray(true_shp)-np.asarray(expected_shp),kernel_size-stride,'\n')
                        assert true_shp >= expected_shp

    # test SAMEpaddingConv
    for x in [torch.zeros(1, 3, 33, 32), torch.zeros(1, 3, 15, 15)]:
        for Conv in [nn.MaxPool2d]:
            for stride in [2, 1]:
                for dilation in [1]:
                    for kernel_size in [3, 5]:
                        conv = Conv(kernel_size=kernel_size, stride=stride, dilation=dilation)
                        # print("kernel_size:",kernel_size, " stride:", stride," trans.:",conv.transposed, " x:", x.size())
                        x_shp = tuple(x.size())
                        conv = SAMEpaddingConv(x_shp, conv)
                        x_shp = x_shp[2:]
                        expected_shp = tuple([t * stride for t in x_shp]) if conv.conv.transposed else tuple(
                            [t // stride for t in x_shp])
                        y = conv(x)
                        true_shp = tuple(y.size())[2:]
                        # print(true_shp,expected_shp,np.asarray(true_shp)-np.asarray(expected_shp),kernel_size-stride,'\n')
                        assert true_shp >= expected_shp
