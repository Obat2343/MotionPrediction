import torch
from einops import rearrange

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, rpe_config='none'):
        """
        from https://github.com/The-AI-Summer/self-attention-cv/blob/main/self_attention_cv/transformer_vanilla/mhsa.py
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = torch.nn.Conv2d(dim, dim*3, 1, 1, 0)
        self.W_0 = torch.nn.Conv2d(_dim, dim, 1, 1, 0)
        self.scale_factor = self.dim_head ** -0.5
        
        if rpe_config != 'none':
            self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config,
                  head_dim=self.dim_head,
                  num_heads=self.heads)
        else:
            self.rpe_q = self.rpe_k = self.rpe_v = None

    def forward(self, x):
        debug_info = {}
        qkv = self.to_qvk(x)  # [batch, dim*3*heads, h, w]
        b, _, h, w = qkv.shape
        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be: [3, batch, heads, tokens, dim_head]
        q, k, v = tuple(rearrange(qkv, 'b (d k m) h w -> k b m (h w) d', k=3, m=self.heads)) # b, heads, tokens, dim_heads
        out, attention = self.compute_mhsa(q, k, v, h, w, scale_factor=self.scale_factor)

        if self.training:
            debug_info["atten_map"] = attention.detach().cpu()
        # Apply final linear transformation layer
        return self.W_0(out), debug_info
    
    def compute_mhsa(self, q, k, v, h, w, scale_factor=1):
        # resulted shape will be: [batch, heads, tokens, tokens]
        dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) # b, heads, tokens, tokens
        
        if self.rpe_k is not None:
            dot_prod += self.rpe_k(q, height=h, width=w)
            
        if self.rpe_q is not None:
            dot_prod += self.rpe_q(k, height=h, width=w)
        
        attention = torch.softmax(dot_prod * scale_factor, dim=-1) # b, heads, tokens, tokens
        
        output = torch.einsum('... i j , ... j d -> ... i d', attention, v) # b, heads, tokens, dim_heads
        if self.rpe_v is not None:
            output += self.rpe_v(attention, height=h, width=w)
        
        output = rearrange(output, 'b m (h w) d -> b (m d) h w', w=w)
        
        # calc result per head
        return output, attention