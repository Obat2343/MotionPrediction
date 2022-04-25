import torch
import torch.nn as nn
from einops import rearrange
from .iRPE.irpe import build_rpe

class DensePatchAttention(nn.Module):
    """
    Attention named Spatial Reduction Attention. See https://paperswithcode.com/method/spatial-reduction-attention.
    """
    def __init__(self, dim, heads=8, dim_head=0, patch_size=8, rpe_config='none'):
        super().__init__()
        
        if dim_head == 0:
            dim_head = dim // heads
        inner_dim = dim_head *  heads
        
        if type(patch_size) == int:
            patch_height = patch_size
            patch_width = patch_size
        elif (type(patch_size) == tuple) or (type(patch_size) == list):
            patch_height = patch_size[0]
            patch_width = patch_size[1]
            
        self.heads = heads

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Conv2d(dim, inner_dim, 1, 1, 0)
        self.to_k = nn.Conv2d(dim, inner_dim, (patch_height, patch_width), (patch_height , patch_width), 0)
        self.to_v = nn.Conv2d(dim, inner_dim, (patch_height, patch_width), (patch_height , patch_width), 0)
        self.softmax = torch.nn.Softmax(dim=3)
        
        if rpe_config != 'none':
            self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config,
                head_dim=dim_head,
                num_heads=heads)
        else:
            self.rpe_q = self.rpe_k = self.rpe_v = None

    def forward(self, x):
        debug_info = {}
        q = self.to_q(x) # B C H W
        k = self.to_k(x) # B C (H/P) (W/P)
        v = self.to_v(x) # B C (H/P) (W/P)
        B,C,H,W = x.shape
        _,_,PH,PW = k.shape
        
        q, k, v = map(lambda t: rearrange(t, 'B (c h) H W -> B h (H W) c', h = self.heads), (q, k, v)) # q: B h c (HW), k: B h c (HW/P**2)
        dot_prod = torch.einsum('bhnc,bhkc->bhnk',q,k) # B h (HW) (HW/P**2)
        
        if self.rpe_k is not None:
            dot_prod += self.rpe_k(q, height=PH, width=PW)
            
        if self.rpe_q is not None:
            dot_prod += self.rpe_q(k, height=PH, width=PW)
            
        atten = self.softmax(dot_prod) # B h (HW) (HW/P**2)
        
        result = torch.einsum('bhnk,bhkc->bhnc',atten,v)
        
        if self.rpe_v is not None:
            result += self.rpe_v(atten, height=PH, width=PW)
            
        result = rearrange(result, 'B h (H W) c -> B (c h) H W', H=H, W=W)
        return result, debug_info