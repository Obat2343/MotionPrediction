import torch
import torch.nn as nn
from timm.models.layers import DropPath

from .Attention import AttentionBlock

class CoordConv2d(torch.nn.modules.conv.Conv2d):
    """
    from https://github.com/walsvid/CoordConv/blob/master/coordconv.py
         https://arxiv.org/pdf/1807.03247.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.with_r = with_r
        self.conv = nn.Conv2d(in_channels + 2 + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out
    
    def addcoords(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        device = input_tensor.device
        batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
        xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

        xx_range = torch.arange(dim_y, dtype=torch.int32)
        yy_range = torch.arange(dim_x, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        # transpose y
        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (dim_y - 1)
        yy_channel = yy_channel.float() / (dim_x - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

        xx_channel = xx_channel.to(device)
        yy_channel = yy_channel.to(device)
        out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out = torch.cat([out, rr], dim=1)

        return out

class ConvLayer(nn.Module):
    
    def __init__(self, input_dim, output_dim, kernel, stride, pad):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel, stride, pad)
        
    def forward(self,x):
        return self.conv(x), {}

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x, None

class ConvBlock(nn.Module):
    r""" 
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, kernel=3, stride=1, padding=1, act='gelu', norm='layer', drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim // 4, 1, 1, 0)
        self.conv1 = nn.Conv2d(dim // 4, dim // 4, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim // 4, dim, 1, 1, 0)
        self.norm = self.norm_layer(dim // 4, name=norm)
        self.act = self.activation_layer(act)
        
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv0(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)
        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1)
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x, None
    
    @staticmethod
    def norm_layer(dim, name='none',num_group=0):
        if name == 'batch':
            layer = nn.BatchNorm2d(dim)
        elif name == 'layer':
            layer = nn.GroupNorm(1, dim)
        elif name == 'instance':
            layer = nn.InstanceNorm2d(dim)
        elif name == 'group':
            assert num_group == 0, "change num_group. Current num_group is 0"
            layer = nn.GroupNorm(num_group, dim)
        elif name == 'none':
            layer = nn.Identity()
        else:
            raise ValueError("Invalid norm")
        return layer
    
    @staticmethod
    def activation_layer(name):
        if name == 'relu':
            layer = nn.ReLU()
        elif name == 'prelu':
            layer = nn.PReLU()
        elif name == 'lrelu':
            layer = nn.LeakyReLU(0.2)
        elif name == 'tanh':
            layer = nn.Tanh()
        elif name == 'sigmoid':
            layer = nn.Sigmoid()
        elif name == 'gelu':
            layer = nn.GELU()
        elif name == 'none':
            layer = nn.Identity()
        else:
            raise ValueError("Invalid activation")
        return layer

class Resnet_Like_Encoder(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, image_size, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], layers=['conv','conv','conv','conv'], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, activation='gelu', norm='layer', atten='swin', heads=4, pos_emb='none', stem_depth=0, stem_layer='conv'):
        super().__init__()
        
        # stem
        self.stem = nn.ModuleList([nn.Conv2d(in_chans, dims[0], 3, 1, 1), self.activation_layer(activation)])
        if stem_depth != 0:
            for i in range(stem_depth):
                self.stem.append(self.build_block(image_size, dims[0], layers[0], activation, norm, 0., 
                layer_scale_init_value, atten, pos_emb, heads, i))
        
        # difine downsample layer
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        down_stem = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=4, stride=4),
            self.norm_layer(dims[0], name=norm)
        )
        image_size = image_size // 4
        
        self.downsample_layers.append(down_stem)
        for i in range(len(depths)-1):
            downsample_layer = nn.Sequential(
                    self.norm_layer(dims[i], name=norm),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        # define main stage
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0 # current layer index
        for i in range(len(depths)):
            stage = nn.ModuleList(
                [self.build_block(image_size // (2**i), dims[i], layers[i], activation, norm, dp_rates[cur + j], 
                layer_scale_init_value, atten, pos_emb, heads, cur + j) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
    
    @staticmethod
    def build_block(image_size, dim, layer_name, activation, norm, drop_path_rate, layer_scale_init_value, atten="swin", pos_emb='none', heads=4, layer_index=0):
        if layer_name == 'conv':
            layer = ConvBlock(dim, act=activation, norm=norm, drop_path=drop_path_rate, layer_scale_init_value=layer_scale_init_value)
        elif layer_name == 'convnext':
            layer = ConvNextBlock(dim, drop_path=drop_path_rate, layer_scale_init_value=layer_scale_init_value)
        elif layer_name == 'atten':
            layer = AttentionBlock(image_size, dim, heads, layer_index, attention=atten, pos_emb=pos_emb, activation=activation, norm=norm,
                        drop_path=drop_path_rate, rel_emb_method='cross', rel_emb_ratio=1.9, rel_emb_mode='ctx', rel_emb_skip=0)
        else:
            raise ValueError("invalid block")
        
        return layer
        
    @staticmethod
    def norm_layer(dim, name='none',num_group=0):
        if name == 'batch':
            layer = nn.BatchNorm2d(dim)
        elif name == 'layer':
            layer = nn.GroupNorm(1, dim)
        elif name == 'instance':
            layer = nn.InstanceNorm2d(dim)
        elif name == 'group':
            assert num_group == 0, "change num_group. Current num_group is 0"
            layer = nn.GroupNorm(num_group, dim)
        elif name == 'none':
            layer = nn.Identity()
        else:
            raise ValueError("Invalid norm")
        return layer
    
    @staticmethod
    def activation_layer(name):
        if name == 'relu':
            layer = nn.ReLU()
        elif name == 'prelu':
            layer = nn.PReLU()
        elif name == 'lrelu':
            layer = nn.LeakyReLU(0.2)
        elif name == 'tanh':
            layer = nn.Tanh()
        elif name == 'sigmoid':
            layer = nn.Sigmoid()
        elif name == 'gelu':
            layer = nn.GELU()
        elif name == 'none':
            layer = nn.Identity()
        else:
            raise ValueError("Invalid activation")
        return layer

    def forward(self, x):
        feature_list = []
        for i in range(len(self.stem)):
            x = self.stem[i](x)
            
        for i in range(len(self.stages)):
            x = self.downsample_layers[i](x)
            for block in self.stages[i]:
                x, _ = block(x)
            feature_list.append(x)
        return feature_list

class Resnet_Like_Decoder(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        out_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, image_size, depths=[3, 3, 3], enc_dims=[96, 192, 384, 768], layers=['conv','conv','conv'], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, activation='gelu', norm='layer', atten='swin', heads=4, pos_emb='none'):
        super().__init__()

        # difine downsample layer
        self.last_upsample = nn.Sequential(
            nn.ConvTranspose2d(enc_dims[0], enc_dims[0] // 2, kernel_size=4, stride=4),
        )
        image_size = image_size // 4
        
        self.upsample_layers = nn.ModuleList()
        for i in range(len(depths)):
            upsample_layer = nn.Sequential(
                    self.norm_layer(enc_dims[i+1], name=norm),
                    nn.ConvTranspose2d(enc_dims[i+1], enc_dims[i+1] // 2, kernel_size=2, stride=2)
            )
            self.upsample_layers.append(upsample_layer)

        # define main stage
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(drop_path_rate, 0, sum(depths))]
        cur = 0 # current layer index
        for i in range(len(depths)):
            stage = nn.ModuleList([ConvLayer(enc_dims[i+1]//2 + enc_dims[i] , enc_dims[i], 1, 1, 0)])
            for j in range(depths[i]):
                stage.append(self.build_block(image_size // (2**i), enc_dims[i], layers[i], activation, norm, dp_rates[cur + j], layer_scale_init_value, atten, pos_emb, heads, cur + j))
            self.stages.append(stage)
            cur += depths[i]

    def forward(self, x):
        feature = x[-1]
        for i in range(len(self.stages)):
            reverse_i = -(i+1)
            feature = self.upsample_layers[reverse_i](feature)
            feature = torch.cat([feature, x[reverse_i - 1]], 1)
            for block in self.stages[reverse_i]:
                feature, _ = block(feature)
        
        feature = self.last_upsample(feature)
        return feature
    
    @staticmethod
    def build_block(image_size, dim, layer_name, activation, norm, drop_path_rate, layer_scale_init_value, atten="swin", pos_emb='none', heads=4, layer_index=0):
        if layer_name == 'conv':
            layer = ConvBlock(dim, act=activation, norm=norm, drop_path=drop_path_rate, layer_scale_init_value=layer_scale_init_value)
        elif layer_name == 'convnext':
            layer = ConvNextBlock(dim, drop_path=drop_path_rate, layer_scale_init_value=layer_scale_init_value)
        elif layer_name == 'atten':
            layer = AttentionBlock(image_size, dim, heads, layer_index, attention=atten, pos_emb=pos_emb, activation=activation, norm=norm,
                        drop_path=drop_path_rate, rel_emb_method='cross', rel_emb_ratio=1.9, rel_emb_mode='ctx', rel_emb_skip=0)
        else:
            raise ValueError("invalid block")
        
        return layer
        
    @staticmethod
    def norm_layer(dim, name='none',num_group=0):
        if name == 'batch':
            layer = nn.BatchNorm2d(dim)
        elif name == 'layer':
            layer = nn.GroupNorm(1, dim)
        elif name == 'instance':
            layer = nn.InstanceNorm2d(dim)
        elif name == 'group':
            assert num_group == 0, "change num_group. Current num_group is 0"
            layer = nn.GroupNorm(num_group, dim)
        elif name == 'none':
            layer = nn.Identity()
        else:
            raise ValueError("Invalid norm")
        return layer
    
    @staticmethod
    def activation_layer(name):
        if name == 'relu':
            layer = nn.ReLU()
        elif name == 'prelu':
            layer = nn.PReLU()
        elif name == 'lrelu':
            layer = nn.LeakyReLU(0.2)
        elif name == 'tanh':
            layer = nn.Tanh()
        elif name == 'sigmoid':
            layer = nn.Sigmoid()
        elif name == 'gelu':
            layer = nn.GELU()
        elif name == 'none':
            layer = nn.Identity()
        else:
            raise ValueError("Invalid activation")
        return layer