import torch.nn as nn
from axial_attention import AxialAttention
        
class AxialAttentionLayer(nn.Module):
    """
    Axial Attention from https://github.com/lucidrains/axial-attention
    """
    def __init__(self, dim, heads):
        """
        Parameters
        ----------
        image_size : int
            Size of images. We assume image is square.
        dim: int
            The number of channel or embedded vector.
        heads: int
            The number of heads.
        """
        super().__init__()
        self.attn = AxialAttention(
            dim = dim,
            heads = heads,
            dim_index = 1
        )
        
    def forward(self, img):
        """
        Parameters
        ----------
        img : torch.tensor [shape: B C H W]
        
        return
        ----------
        x : torch.tensor [shape: B C H W]
        """
        debug_info = {}
        img = self.attn(img)
        return img, debug_info