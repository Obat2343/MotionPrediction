import torch
import torch.nn as nn
from timm.models.layers import DropPath
from axial_attention import AxialPositionalEmbedding

from .Normal import MultiHeadSelfAttention
from .Axial import AxialAttentionLayer
from .Patch import DensePatchAttention
from .Swin import SwinTransformerLayer
from .PosEmb import AbsolutePositionalEncodingSin1D_Image, LearnableAbsolutePositionalEncodingSin1D_Image
from .iRPE.irpe import get_rpe_config

class AttentionBlock(nn.Module):
    """
    Attention Block
    Axial Attention from https://github.com/lucidrains/axial-attention
    """
    def __init__(self, image_size, dim, heads, layer_index, attention='mha', pos_emb='abs', activation='gelu', norm='layer', feedforward='mlp',
                        drop=0., drop_path=0., rel_emb_method='cross', rel_emb_ratio=1.9, rel_emb_mode='ctx', rel_emb_skip=0):
        """
        Parameters
        ----------
        image_size : int
            Size of images. We assume image is square.
        dim: int
            The number of channel or embedded vector.
        heads: int
            The number of heads.
        layer_index: int
            Index of this block. This index is only used for the swin.
        attention: str
            Name of attention. 'mha', 'axial', 'patch' and 'swin' are avaiable.
        pos_emb: str
            Name of position embedding. 'abs', 'labs', 'axial', 'iRPE' and 'none' are available. 
            Please use none for swin transformer because it includes the unique embedding method. 
            Also, please use axial for axial transformer due to the same reason.
            iRPE is avaiable for 'mha' and 'patch'
        activation: str
            Name of actionvation. gelu is default.
        norm: str
            Name of normalization. This block has two norm layer and use same norm method.
        feedforward: str
            Network after the attention calculation. mlp is basic.
        drop: float
            Dropout rate for mlp(feedforward net).
        drop_path: float
            Drop path rate.
        rel_emb_method: str
            Choice of iRPE method. e.g. 'euc', 'quant', 'cross', 'product'
        rel_emb_method: int (0 or 1)
            The number of skip token before spatial tokens.
            When skip is 0, no classification token.
            When skip is 1, there is a classification token before spatial tokens.
        """
        super().__init__()
        
        self.image_size = image_size
        self.dim = dim
        self.heads = heads
        self.layer_index = layer_index
        self.drop = drop
        
        self.pos_emb = self.posemb_layer(dim,image_size,pos_emb)
        if pos_emb == 'iRPE':
            self.rpe_config = get_rpe_config(
                        rel_emb_ratio=rel_emb_ratio,
                        rel_emb_method=rel_emb_method,
                        rel_emb_mode=rel_emb_mode,
                        shared_head=True,
                        skip=rel_emb_skip,
                        rpe_on='qkv',
                        )
        else:
            self.rpe_config = 'none'
            
        self.attn = self.attention_layer(attention)
        self.prenorm = self.norm_layer(dim, norm)
        self.feedforward = self.feedforward_layer(feedforward,activation)
        self.postnorm = self.norm_layer(dim, norm)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.tensor [shape: B C H W]
        
        return
        ----------
        x : torch.tensor [shape: B C H W]
        """
        # add pos_emb
        x = self.pos_emb(x)
        
        # attntion
        attn = self.prenorm(x)
        attn, debug_info = self.attn(attn)
        x = x + self.drop_path(attn)
        
        # feedforward
        ff = self.postnorm(x)
        ff = self.feedforward(ff)
        x = x + self.drop_path(ff)
        return x, debug_info
    
    def attention_layer(self,name):
        if name == 'mha':
            layer = MultiHeadSelfAttention(self.dim,self.heads,self.rpe_config)
        elif name == 'axial':
            layer = AxialAttentionLayer(self.dim,self.heads)
        elif name == 'patch':
            layer = DensePatchAttention(self.dim,self.heads,patch_size=8)
        elif name == 'swin':
            shift = (self.layer_index % 2 != 0)
            layer = SwinTransformerLayer(self.dim,(self.image_size, self.image_size), self.heads,
                                         window_size=8, shift=shift, attn_drop=self.drop)
        elif name == 'none':
            layer = nn.Identity()
        else:
            raise ValueError("Invalid attention")
        return layer
    
    @staticmethod
    def posemb_layer(dim,image_size,name):
        if name == 'abs':
            layer = AbsolutePositionalEncodingSin1D_Image(dim,max_tokens=image_size*image_size)
        elif name == 'labs':
            layer = LearnableAbsolutePositionalEncodingSin1D_Image(dim)
        elif name == 'axial':
            layer = AxialPositionalEmbedding(dim=dim,shape=(image_size, image_size))
        elif name == 'none':
            layer = nn.Identity()
        else:
            raise ValueError("Invalid pos_emb")
        return layer
    
    def feedforward_layer(self,name,activation='gelu'):
        if name == 'mlp':
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(self.dim, self.dim, 1, 1, 0),
                self.activation_layer(activation),
                torch.nn.Dropout(self.drop),
                torch.nn.Conv2d(self.dim, self.dim, 1, 1, 0),
                torch.nn.Dropout(self.drop))
        elif name == 'conv3':
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(self.dim, self.dim, 3, 1, 1),
                self.activation_layer(activation),
                torch.nn.Dropout(self.drop),
                torch.nn.Conv2d(self.dim, self.dim, 3, 1, 1),
                torch.nn.Dropout(self.drop))
        elif name == 'none':
            layer = nn.Identity()
        else:
            raise ValueError("Invalid feedforward")
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
    