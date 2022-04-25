import torch
import torch.nn as nn
from einops import rearrange, repeat

def expand_to_batch(tensor, desired_size):
    tile = desired_size // tensor.shape[0]
    return repeat(tensor, 'b ... -> (b tile) ...', tile=tile)

# adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class AbsolutePositionalEncodingSin1D(nn.Module):
    """
    from https://github.com/The-AI-Summer/self-attention-cv/tree/main/self_attention_cv
    This function is used in the original Transformer.
    """
    def __init__(self, dim, dropout=0.1, max_tokens=5000):
        super(AbsolutePositionalEncodingSin1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(1, max_tokens, dim)
        position = torch.arange(0, max_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.Tensor([max_tokens*2])) / dim))
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = pe

    def forward(self, x):
        batch, seq_tokens, _ = x.size()
        device = x.device
        x = x + expand_to_batch(self.pe[:, :seq_tokens, :], desired_size=batch).to(device)
        return self.dropout(x)

# adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class AbsolutePositionalEncodingSin1D_Image(AbsolutePositionalEncodingSin1D):

    def forward(self, x):
        b, c, h, w = x.size()
        device = x.device
        position_emb = expand_to_batch(self.pe[:,:h*w,:], desired_size=b)
        position_emb = rearrange(position_emb, 'b (h w) p -> b p h w', h=h, w=w)
        x = x + position_emb.to(device)
        return self.dropout(x)

class AbsolutePositionalEncodingSin2D_debug_x(AbsolutePositionalEncodingSin1D):
    
    def forward(self, x):
        b, c, h, w = x.size()
        position_emb_x = expand_to_batch(self.pe[:,:w,:], desired_size=b)
        position_emb_y = expand_to_batch(self.pe[:,:h,:], desired_size=b)
        
        position_emb_x = repeat(position_emb_x, 'b w p -> b p w h', h=h)
        x = x + (position_emb_x)
        return self.dropout(x)

class AbsolutePositionalEncodingSin2D_debug_y(AbsolutePositionalEncodingSin1D):
    
    def forward(self, x):
        b, c, h, w = x.size()
        position_emb_x = expand_to_batch(self.pe[:,:w,:], desired_size=b)
        position_emb_y = expand_to_batch(self.pe[:,:h,:], desired_size=b)
        
        position_emb_y = repeat(position_emb_y, 'b h p -> b p w h', w=w)
        x = x + (position_emb_y)
        return self.dropout(x)

class AbsolutePositionalEncodingSin2D(AbsolutePositionalEncodingSin1D):
    
    def __init__(self, dim, dropout=0.1, max_tokens=5000):
        super(AbsolutePositionalEncodingSin1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pex = torch.zeros(1, max_tokens, dim)
        pey = torch.zeros(1, max_tokens, dim)
        position = torch.arange(0, max_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.Tensor([max_tokens*2])) / dim))
        pex[..., 0::2] = torch.sin(position * div_term)
        pex[..., 1::2] = torch.cos(position * div_term)
        pey[..., 0::2] = torch.sin(position * div_term)
        pey[..., 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        self.pex = pex
        self.pey = pey
        
    def forward(self, x):
        b, c, h, w = x.size()
        position_emb_x = expand_to_batch(self.pex[:,:w,:], desired_size=b)
        position_emb_y = expand_to_batch(self.pey[:,:h,:], desired_size=b)
        
        position_emb_x = repeat(position_emb_x, 'b w p -> b p w h', h=h)
        position_emb_y = repeat(position_emb_y, 'b h p -> b p w h', w=w)
        x = x + (position_emb_x) + (position_emb_y)
        return self.dropout(x)

class LearnableAbsolutePositionalEncodingSin1D(nn.Module):
    """
    from https://github.com/The-AI-Summer/self-attention-cv/tree/main/self_attention_cv
    This function is used in the original Transformer.
    """
    def __init__(self, dim, dropout=0.1):
        super(LearnableAbsolutePositionalEncodingSin1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        self.div_term = nn.Parameter(torch.rand(1, int(self.dim / 2)), requires_grad=True)

    def forward(self, x):
        batch, seq_tokens, dim = x.size()
        device = x.device
        
        assert dim == self.dim, f'expected dim is {self.dim} but the input dim is {dim}.'
        position = torch.arange(0, seq_tokens, dtype=torch.float).unsqueeze(1)
        position = repeat(position, 't d -> b t d', b=batch).to(device)
        pe = torch.cat((torch.sin(position * self.div_term),torch.cos(position * self.div_term)), 2)
        x = x + pe
        
        return self.dropout(x)

# adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class LearnableAbsolutePositionalEncodingSin1D_Image(LearnableAbsolutePositionalEncodingSin1D):

    def forward(self, x):
        b, c, h, w = x.size()
        device = x.device
        seq_tokens = h * w
        
        assert c == self.dim, f'expected dim is {self.dim} but the input dim is {dim}.'
        position = torch.arange(0, seq_tokens, dtype=torch.float).unsqueeze(1)
        position = repeat(position, 't d -> b t d', b=b).to(device)
        pe = torch.cat((torch.sin(position * self.div_term),torch.cos(position * self.div_term)), 2)
        pe = rearrange(pe, 'b (h w) p -> b p h w', h=h, w=w)
        x = x + pe
        
        return self.dropout(x)
    
class LearnableAbsolutePositionalEncodingSin2D(nn.Module):
    """
    from https://github.com/The-AI-Summer/self-attention-cv/tree/main/self_attention_cv
    This function is used in the original Transformer.
    """
    def __init__(self, dim, dropout=0.1):
        super(LearnableAbsolutePositionalEncodingSin2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        self.div_term_x = nn.Parameter(torch.rand(1, int(self.dim / 2)), requires_grad=True)
        self.div_term_y = nn.Parameter(torch.rand(1, int(self.dim / 2)), requires_grad=True)
        
    def forward(self, x):
        b, c, h, w = x.size()
    
        assert c == self.dim, f'expected dim is {self.dim} but the input dim is {dim}.'
        position_x = torch.arange(0, w, dtype=torch.float).unsqueeze(1)
        position_x = repeat(position_x, 't d -> b t d', b=b)
        pe_x = torch.cat((torch.sin(position_x * self.div_term_x),torch.cos(position_x * self.div_term_x)), 2)
        pe_x = repeat(pe_x, 'b w p -> b p w h', h=h)
        
        position_y = torch.arange(0, h, dtype=torch.float).unsqueeze(1)
        position_y = repeat(position_y, 't d -> b t d', b=b)
        pe_y = torch.cat((torch.sin(position_y * self.div_term_y),torch.cos(position_y * self.div_term_y)), 2)
        pe_y = repeat(pe_y, 'b h p -> b p w h', w=w)
        x = x + pe_x + pe_y
        
        return self.dropout(x)