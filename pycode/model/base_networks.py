import torch
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import time
from .deform_conv import th_batch_map_offsets, th_generate_grid

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SELayer(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LinearBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, activation='prelu', norm=None):
        super(LinearBlock, self).__init__()

        # self.input_size = input_size
        # self.output_size = output_size
        self.linear = torch.nn.Linear(input_size, output_size)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_size)
        elif self.norm == 'spectral':
            self.conv = torch.nn.utils.spectral_norm(self.conv)
        elif self.norm == 'none':
            self.norm = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(False)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, False)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif self.activation == 'none':
            self.activation = None
    
    def forward(self, x):
        if (self.norm is not None) and (self.norm != 'spectral'):
            out = self.bn(self.linear(x))
        else:
            out = self.linear(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()

        # self.input_size = input_size
        # self.output_size = output_size

        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, dilation, groups, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_size)
        elif self.norm == 'spectral':
            self.conv = torch.nn.utils.spectral_norm(self.conv)
        elif self.norm == 'none':
            self.norm = None
        
        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU()
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif self.activation == 'none':
            self.activation = None

    def forward(self, x):
        if (self.norm is not None) and (self.norm != 'spectral'):
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class ConvBlock_Pre(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, activation='prelu', norm=None):
        # ConvBlock Pre activation
        super(ConvBlock_Pre, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, dilation, groups, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_size)
        elif self.norm == 'spectral':
            self.conv = torch.nn.utils.spectral_norm(self.conv)
        elif self.norm == 'none':
            self.norm = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(False)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, False)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.activation is not None:
            x = self.act(x)

        if (self.norm is not None) and (self.norm != 'spectral'):
            return self.bn(self.conv(x))
        else:
            return self.conv(x)

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_size)
        elif self.norm == 'spectral':
            self.deconv = torch.nn.utils.spectral_norm(self.deconv)
        elif self.norm == 'none':
            self.norm = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(False)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, False)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if (self.norm is not None) and (self.norm != 'spectral'):
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class PSBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, scale_factor, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(PSBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size * scale_factor**2, kernel_size, stride, padding, bias=bias)
        self.ps = torch.nn.PixelShuffle(scale_factor)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_size)
        elif self.norm == 'spectral':
            self.conv = torch.nn.utils.spectral_norm(self.conv)
        elif self.norm == 'none':
            self.norm = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if (self.norm is not None) and (self.norm != 'spectral'):
            out = self.bn(self.ps(self.conv(x)))
        else:
            out = self.ps(self.conv(x))

        if self.activation is not None:
            out = self.act(out)
        return out

class RConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(RConvBlock, self).__init__()
        
        self.up = torch.nn.Upsample(scale_factor=stride, mode='bilinear')
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size-1, 1, kernel_size-(2*padding)-1, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_size)
        elif self.norm == 'spectral':
            self.conv = torch.nn.utils.spectral_norm(self.deconv)
        elif self.norm == 'none':
            self.norm = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()


    def forward(self, x):
        # x = self.up(x)
        if (self.norm is not None) and (self.norm != 'spectral'):
            out = self.bn(self.conv(self.up(x)))
        else:
            out = self.conv(self.up(x))

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class ConvOffset2D(torch.nn.Conv2d):
    """ConvOffset2D
    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation
    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage

    from https://github.com/oeway/pytorch-deform-conv/blob/master/torch_deform_conv/layers.py
    """
    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init
        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """

        self.filters = filters
        self._grid_param = None
        super(ConvOffset2D, self).__init__(self.filters, self.filters*2, 3, padding=1, bias=False, **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))

    def forward(self, x):
        """Return the deformed featured map"""
        x_shape = x.size()
        offsets = super(ConvOffset2D, self).forward(x)

        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)

        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # X_offset: (b*c, h, w)
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(x))

        # x_offset: (b, h, w, c)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)

        return x_offset

    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(1), x.size(2)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_height, input_width, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x

class BasicRFB(torch.nn.Module):

    def __init__(self, in_planes, out_planes, activation='relu', norm=None, scale = 0.1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = torch.nn.Sequential(
                ConvBlock(in_planes, 2*inter_planes, 1, 1, 0, activation=activation, norm=norm),
                ConvBlock(2*in_planes, 2*inter_planes, 3, 1, 1, activation=None, norm=norm)
                )
        self.branch1 = torch.nn.Sequential(
                ConvBlock(in_planes, inter_planes, 1, 1, activation=activation, norm=norm),
                ConvBlock(inter_planes, 2*inter_planes, 3, 1, 1, activation=activation, norm=norm),
                ConvBlock(2*inter_planes, 2*inter_planes, 3, 1, 2, 2, activation=None, norm=norm)
                )
        self.branch2 = torch.nn.Sequential(
                ConvBlock(in_planes, inter_planes, 1, 1, activation=activation, norm=norm),
                ConvBlock(inter_planes, (inter_planes//2)*3, 3, 1, 1, activation=activation, norm=norm),
                ConvBlock((inter_planes//2)*3, 2*inter_planes, 3, 1, 1, activation=activation, norm=norm),
                ConvBlock(2*inter_planes, 2*inter_planes, 3, 1, 3, 3, activation=None, norm=norm)
                )

        self.ConvLinear = ConvBlock(6*inter_planes, out_planes, 1, 1, activation=None, norm=norm)
        self.shortcut = ConvBlock(in_planes, out_planes, 1, 1, activation=None, norm=norm)
        
        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short

        if self.activation is not None:
            out = self.act(out)

        return out

class InceptionDownBlock(torch.nn.Module):
    def __init__(self, in_channels, activation=None, norm=None):
        super(InceptionDownBlock, self).__init__()
        self.branch8x8_1 = ConvBlock(in_channels, in_channels // 4, 1, 1, 0, activation=None)
        self.branch8x8_2 = ConvBlock(in_channels // 4, in_channels // 4, 8, 4, 2, activation=None)

        self.branch4x4_1 = ConvBlock(in_channels, in_channels // 4, 1, 1, 0, activation=None)
        self.branch4x4_2 = ConvBlock(in_channels // 4, in_channels // 4, 4, 2, 1, activation=None)
        self.branch4x4_3 = ConvBlock(in_channels // 4, in_channels // 4, 4, 2, 1, activation=None)

        self.branch2x2_1 = ConvBlock(in_channels, in_channels // 4, 1, 1, 0, activation=None)
        self.branch2x2_2 = ConvBlock(in_channels // 4, in_channels // 4, 2, 2, 0, activation=None)
        self.branch2x2_3 = ConvBlock(in_channels // 4, in_channels // 4, 2, 2, 0, activation=None)

        self.branch_pool = ConvBlock(in_channels, in_channels // 4, 1, 1, 0, activation=None)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_size)
        elif self.norm == 'spectral':
            self.conv = torch.nn.utils.spectral_norm(self.conv)
        elif self.norm == 'none':
            self.norm = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        branch8x8 = self.branch8x8_1(x)
        branch8x8 = self.branch8x8_2(branch8x8)

        branch4x4 = self.branch4x4_1(x)
        branch4x4 = self.branch4x4_2(branch4x4)
        branch4x4 = self.branch4x4_3(branch4x4)

        branch2x2 = self.branch2x2_1(x)
        branch2x2 = self.branch2x2_2(branch2x2)
        branch2x2 = self.branch2x2_3(branch2x2)

        branch_pool = self.branch_pool(x)
        branch_pool = F.avg_pool2d(branch_pool, kernel_size=8, stride=4, padding=2)

        outputs = [branch8x8, branch4x4, branch2x2, branch_pool]
        outputs = torch.cat(outputs, 1)

        if (self.norm is not None) and (self.norm != 'spectral'):
            outputs = self.bn(outputs)

        if self.activation is not None:
            outputs = self.act(outputs)

        return outputs

class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None, up_mode='deconv', se_block=False, inception=False):
        super(UpBlock, self).__init__()
            
        print(up_mode)
        if up_mode == 'deconv':
            self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
            self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        elif up_mode == 'rconv':
            self.up_conv1 = RConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
            self.up_conv3 = RConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        elif up_mode == 'pixelshuffle':
            self.up_conv1 = PSBlock(num_filter, num_filter, 4, 3, 1, 1, bias=bias, activation=activation, norm=norm)
            self.up_conv3 = PSBlock(num_filter, num_filter, 4, 3, 1, 1, bias=bias, activation=activation, norm=norm)
        else:
            raise ValueError()

        if inception == True:
            self.down_conv = InceptionDownBlock(num_filter, activation=activation, norm=norm)
        else:
            self.down_conv = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)

        if se_block == True:
            self.se = SELayer(num_filter, 8)
        else:
            self.se = None

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.down_conv(h0)
        h1 = self.up_conv3(l0 - x)
        if self.se != None:
            return self.se(h1 + h0)
        else:
            return h1 + h0

class D_UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None, up_mode='deconv', se_block=False, inception=False):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, bias=bias, activation=activation, norm=norm)

        if up_mode == 'pixelshuffle':
            self.up_conv1 = PSBlock(num_filter, num_filter, 4, 3, 1, 1, bias=bias, activation=activation, norm=norm)
            self.up_conv3 = PSBlock(num_filter, num_filter, 4, 3, 1, 1, bias=bias, activation=activation, norm=norm)
        elif up_mode == 'deconv':
            self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
            self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        elif up_mode == 'rconv':
            self.up_conv1 = RConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
            self.up_conv3 = RConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        else:
            raise ValueError()

        if inception == True:
            self.down_conv = InceptionDownBlock(num_filter, activation=activation, norm=norm)
        else:
            self.down_conv = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)

        if se_block == True:
            self.se = SELayer(num_filter, 8)
        else:
            self.se = None

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.down_conv(h0)
        h1 = self.up_conv3(l0 - x)
        if self.se != None:
            return self.se(h1 + h0)
        else:
            return h1 + h0

class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None, up_mode='deconv', se_block=False, inception=False):
        super(DownBlock, self).__init__()

        if se_block == True:
            self.se = SELayer(num_filter, 8)
        else:
            self.se = None

        if inception == True:
            self.down_conv1 = InceptionDownBlock(num_filter, activation=activation, norm=norm)
            self.down_conv3 = InceptionDownBlock(num_filter, activation=activation, norm=norm)
        else:
            self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
            self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)

        if up_mode == 'pixelshuffle':
            self.up_conv = PSBlock(num_filter, num_filter, 4, 3, 1, 1, bias=bias, activation=activation, norm=norm)
        elif up_mode == 'deconv':
            self.up_conv = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        elif up_mode == 'rconv':
            self.up_conv = RConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)


    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.up_conv(l0)
        l1 = self.down_conv3(h0 - x)
        if self.se != None:
            return self.se(l1 + l0)
        else:
            return l1 + l0

class D_DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None, up_mode='deconv', se_block=False, inception=False):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, bias=bias, activation=activation, norm=norm)
        
        if se_block == True:
            self.se = SELayer(num_filter, 8)
        else:
            self.se = None

        if inception == True:
            self.down_conv1 = InceptionDownBlock(num_filter, activation=activation, norm=norm)
            self.down_conv3 = InceptionDownBlock(num_filter, activation=activation, norm=norm)
        else:
            self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
            self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)

        if up_mode == 'pixelshuffle':
            self.up_conv = PSBlock(num_filter, num_filter, 4, 3, 1, 1, bias=bias, activation=activation, norm=norm)
        elif up_mode == 'deconv':
            self.up_conv = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        elif up_mode == 'rconv':
            self.up_conv = RConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)


    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.up_conv(l0)
        l1 = self.down_conv3(h0 - x)
        if self.se != None:
            return self.se(l1 + l0)
        else:
            return l1 + l0

class UNetBlock(torch.nn.Module):
    def __init__(self, base_filter=64, activation=None, norm=None):
        super(UNetBlock, self).__init__()

        if norm == 'none':
            norm = None

        self.conv_blocks = torch.nn.ModuleList([
            ConvBlock(base_filter*1, base_filter*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*1, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*2, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*2, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*4, 3, 1, 1, activation=activation, norm=norm),
        ])
        self.deconv_blocks = torch.nn.ModuleList([
            DeconvBlock(base_filter*4, base_filter*2, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            DeconvBlock(base_filter*2, base_filter*2, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            DeconvBlock(base_filter*2, base_filter*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*1, 3, 1, 1, activation=activation, norm=norm),
        ])

    def forward(self, x):
        sources = [] # 1 2 2
        for i in range(len(self.conv_blocks)):
            if i % 2 == 0 and i != len(self.conv_blocks)-1 :
                sources.append(x)
            x = self.conv_blocks[i](x)

        for i in range(len(self.deconv_blocks)):
            x = self.deconv_blocks[i](x)
            if i % 2 == 0 and len(sources) != 0:
                x = torch.cat((x, sources.pop(-1)), 1)
        
        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim,activation='relu',norm='batch'):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(input_dim, int(output_dim / 2), activation=activation, norm=norm)
        self.conv2 = ConvBlock(int(output_dim / 2), int(output_dim / 2), activation=activation, norm=norm)
        self.conv3 = ConvBlock(int(output_dim / 2), output_dim, activation=activation, norm=norm)
        self.skip_conv = ConvBlock(input_dim, output_dim, 1, 1, 0, activation='none', norm='none')

    def forward(self, x):
        residual = self.skip_conv(x)
        output1 = self.conv1(x)
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        output4 = output3 + residual
        return output4

class hourglass_module(torch.nn.Module):
    """
    paper: https://arxiv.org/pdf/1603.06937.pdf

    code reference:
    https://github.com/Naman-ntc/Pytorch-Human-Pose-Estimation/blob/master/models/StackedHourGlass.py
    https://github.com/princeton-vl/pytorch_stacked_hourglass/blob/master/models/posenet.py

    options:
    activation: relu, prelu, lrelu, tanh, sigmoid
    norm: none, batch, instance, group, spectral
    upsample_mode: nearest, linear, bilinear, bicubic, trilinear
    """

    def __init__(self, input_dim, num_downscale, activation='relu',norm='batch',upsample_mode='nearest'):
        super(hourglass_module, self).__init__()
        self.num_downscale = num_downscale
        self.convblock1 = ResidualBlock(input_dim, input_dim, activation=activation, norm=norm)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.low_block1 = ResidualBlock(input_dim, input_dim, activation=activation, norm=norm)

        if self.num_downscale > 1:
            self.low_block2 = hourglass_module(input_dim, num_downscale-1, activation=activation, norm=norm)
        else:
            self.low_block2 = ResidualBlock(input_dim, input_dim, activation=activation, norm=norm)

        self.low_block3 = ResidualBlock(input_dim, input_dim, activation=activation, norm=norm)
        self.upscale = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
    
    def forward(self, x):
        output1 = self.convblock1(x)
        output2_pool = self.pool(x)
        output2_low1 = self.low_block1(output2_pool)
        output2_low2 = self.low_block2(output2_low1)
        output2_low3 = self.low_block3(output2_low2)
        output2_up = self.upscale(output2_low3)
        return output1 + output2_up

class SoftArgmax2D(torch.nn.Module):
    """
    https://github.com/MWPainter/cvpr2019/blob/master/stitched/soft_argmax.py
    Implementation of a 1d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations.
    """
    def __init__(self, base_index=0, step_size=1, softmax_temp=1.0, device='cpu'):
        """
        The "arguments" are base_index, base_index+step_size, base_index+2*step_size, ... and so on for
        arguments at indices 0, 1, 2, ....
        Assumes that the input to this layer will be a batch of 3D tensors (so a 4D tensor).
        For input shape (B, C, W, H), we apply softmax across the W and H dimensions.
        We use a softmax, over dim 2, expecting a 3D input, which is created by reshaping the input to (B, C, W*H)
        (This is necessary because true 2D softmax doesn't natively exist in PyTorch...
        :param base_index: Remember a base index for 'indices' for the input
        :param step_size: Step size for 'indices' from the input
        :param window_function: Specify window function, that given some center point produces a window 'landscape'. If
            a window function is specified then before applying "soft argmax" we multiply the input by a window centered
            at the true argmax, to enforce the input to soft argmax to be unimodal. Window function should be specified
            as one of the following options: None, "Parzen", "Uniform"
        :param window_width: How wide do we want the window to be? (If some point is more than width/2 distance from the
            argmax then it will be zeroed out for the soft argmax calculation, unless, window_fn == None)
        """
        super(SoftArgmax2D, self).__init__()
        self.base_index = base_index
        self.step_size = step_size
        self.softmax = torch.nn.Softmax(dim=2)
        self.softmax_temp = Parameter(torch.ones(1)*softmax_temp)
        self.device = device



    def _softmax_2d(self, x, temp):
        """
        For the lack of a true 2D softmax in pytorch, we reshape each image from (C, W, H) to (C, W*H) and then
        apply softmax, and then restore the original shape.
        :param x: A 4D tensor of shape (B, C, W, H) to apply softmax across the W and H dimensions
        :param temp: A scalar temperature to apply as part of the softmax function
        :return: Softmax(x, dims=(2,3))
        """
        B, C, W, H = x.size()
        x_flat = x.view((B, C, W*H)) / temp
        x_softmax = self.softmax(x_flat)
        return x_softmax.view((B, C, W, H))


    def forward(self, x):
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:
        SoftArgMax2d(x) = (\sum_i \sum_j (i * softmax2d(x)_ij), \sum_i \sum_j (j * softmax2d(x)_ij))
        :param x: The input to the soft arg-max layer
        :return: Output of the 2D soft arg-max layer, x_coords and y_coords, in the shape (B, C, 2), which are the soft
            argmaxes per channel
        """
        batch_size, channels, height, width = x.size()
        
        # comupute argmax
        argmax = torch.argmax(x.view(batch_size * channels, -1), dim=1)
        argmax_x, argmax_y = torch.remainder(argmax, width).float(), torch.floor(torch.div(argmax.float(), float(width)))
        argmax = torch.cat((argmax_x.view(batch_size, channels, -1), (argmax_y.view(batch_size, channels, -1))), 2)
        
        smax = self._softmax_2d(x, self.softmax_temp)

        # compute x index (sum over y axis, produce with indices and then sum over x axis for the expectation)
        x_end_index = self.base_index + width * self.step_size
        x_indices = torch.arange(start=self.base_index, end=x_end_index, step=self.step_size).to(self.device)
        x_coords = torch.sum(torch.sum(smax, 2) * x_indices, 2)

        # compute y index (sum over x axis, produce with indices and then sum over y axis for the expectation)
        y_end_index = self.base_index + height * self.step_size
        y_indices = torch.arange(start=self.base_index, end=y_end_index, step=self.step_size).to(self.device)
        y_coords = torch.sum(torch.sum(smax, 3) * y_indices, 2)
        
        softargmax = torch.cat([torch.unsqueeze(x_coords, 2), torch.unsqueeze(y_coords, 2)], dim=2)
        # For debugging (testing if it's actually like the argmax?)
        # argmax_x = argmax_x.view(batch_size, channels)
        # argmax_y = argmax_y.view(batch_size, channels)
        # print("X err in soft argmax: {err}".format(err=torch.mean(torch.abs(argmax_x - x_coords))))
        # print("Y err in soft argmax: {err}".format(err=torch.mean(torch.abs(argmax_y - y_coords))))
        
        # Put the x coords and y coords (shape (B,C)) into an output with shape (B,C,2)
        return softargmax, argmax, smax
    
class SigmoidArgmax2D(torch.nn.Module):
    """
    Implementation of a 1d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations.
    """
    def __init__(self, base_index=0, step_size=1, softmax_temp=1.0, device='cpu'):
        """
        The "arguments" are base_index, base_index+step_size, base_index+2*step_size, ... and so on for
        arguments at indices 0, 1, 2, ....
        Assumes that the input to this layer will be a batch of 3D tensors (so a 4D tensor).
        For input shape (B, C, W, H), we apply softmax across the W and H dimensions.
        We use a softmax, over dim 2, expecting a 3D input, which is created by reshaping the input to (B, C, W*H)
        (This is necessary because true 2D softmax doesn't natively exist in PyTorch...
        :param base_index: Remember a base index for 'indices' for the input
        :param step_size: Step size for 'indices' from the input
        :param window_function: Specify window function, that given some center point produces a window 'landscape'. If
            a window function is specified then before applying "soft argmax" we multiply the input by a window centered
            at the true argmax, to enforce the input to soft argmax to be unimodal. Window function should be specified
            as one of the following options: None, "Parzen", "Uniform"
        :param window_width: How wide do we want the window to be? (If some point is more than width/2 distance from the
            argmax then it will be zeroed out for the soft argmax calculation, unless, window_fn == None)
        """
        super(SigmoidArgmax2D, self).__init__()
        self.base_index = base_index
        self.step_size = step_size
        self.sigmoid = torch.nn.Sigmoid()
        self.device = device

    def forward(self, x):
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:
        SoftArgMax2d(x) = (\sum_i \sum_j (i * softmax2d(x)_ij), \sum_i \sum_j (j * softmax2d(x)_ij))
        :param x: The input to the soft arg-max layer
        :return: Output of the 2D soft arg-max layer, x_coords and y_coords, in the shape (B, C, 2), which are the soft
            argmaxes per channel
        """
        batch_size, channels, height, width = x.size()
        
        # comupute argmax
        argmax = torch.argmax(x.view(batch_size * channels, -1), dim=1)
        argmax_x, argmax_y = torch.remainder(argmax, width).float(), torch.floor(torch.div(argmax.float(), float(width)))
        argmax = torch.cat((argmax_x.view(batch_size, channels, -1), (argmax_y.view(batch_size, channels, -1))), 2)
        
        smax = self.sigmoid(x)
        sum_value = torch.sum(smax.view(batch_size, channels, -1), 2) # B, C
        sum_value = torch.cat((torch.unsqueeze(sum_value,-1),torch.unsqueeze(sum_value,-1)),-1)

        # compute x index (sum over y axis, produce with indices and then sum over x axis for the expectation)
        x_end_index = self.base_index + width * self.step_size
        x_indices = torch.arange(start=self.base_index, end=x_end_index, step=self.step_size).to(self.device)
        x_coords = torch.sum(torch.sum(smax, 2) * x_indices, 2)

        # compute y index (sum over x axis, produce with indices and then sum over y axis for the expectation)
        y_end_index = self.base_index + height * self.step_size
        y_indices = torch.arange(start=self.base_index, end=y_end_index, step=self.step_size).to(self.device)
        y_coords = torch.sum(torch.sum(smax, 3) * y_indices, 2)
        
        softargmax = torch.cat([torch.unsqueeze(x_coords, 2), torch.unsqueeze(y_coords, 2)], dim=2)
        softargmax = softargmax / sum_value
        # For debugging (testing if it's actually like the argmax?)
        # argmax_x = argmax_x.view(batch_size, channels)
        # argmax_y = argmax_y.view(batch_size, channels)
        # print("X err in soft argmax: {err}".format(err=torch.mean(torch.abs(argmax_x - x_coords))))
        # print("Y err in soft argmax: {err}".format(err=torch.mean(torch.abs(argmax_y - y_coords))))
        
        # Put the x coords and y coords (shape (B,C)) into an output with shape (B,C,2)
        return softargmax, argmax, smax