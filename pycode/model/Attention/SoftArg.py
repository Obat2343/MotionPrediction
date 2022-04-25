import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn.parameter import Parameter

class SoftArgmax2D(torch.nn.Module):
    """
    https://github.com/MWPainter/cvpr2019/blob/master/stitched/soft_argmax.py
    Implementation of a 1d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations.
    """
    def __init__(self, base_index=0, step_size=1, temp=1.0, activate="softmax"):
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
        if activate == 'softmax':
            self.activate = torch.nn.Softmax(dim=2)
        elif activate == 'sigmoid':
            self.activate = torch.nn.Sigmoid()
        self.temp = Parameter(torch.ones(1)*temp)

    def _activate_2d(self, x, temp):
        """
        For the lack of a true 2D softmax in pytorch, we reshape each image from (C, W, H) to (C, W*H) and then
        apply softmax, and then restore the original shape.
        :param x: A 4D tensor of shape (B, C, W, H) to apply softmax across the W and H dimensions
        :param temp: A scalar temperature to apply as part of the softmax function
        :return: Softmax(x, dims=(2,3))
        """
        B, C, W, H = x.size()
        x_flat = x.view((B, C, W*H)) / temp
        x_activate = self.activate(x_flat)
        return x_activate.view((B, C, W, H))


    def forward(self, x):
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:
        SoftArgMax2d(x) = (\sum_i \sum_j (i * softmax2d(x)_ij), \sum_i \sum_j (j * softmax2d(x)_ij))
        :param x: The input to the soft arg-max layer
        :return: Output of the 2D soft arg-max layer, x_coords and y_coords, in the shape (B, C, 2), which are the soft
            argmaxes per channel
        """
        device = x.device
        batch_size, channels, height, width = x.size()
        
        # comupute argmax
        argmax = torch.argmax(x.view(batch_size * channels, -1), dim=1)
        argmax_x, argmax_y = torch.remainder(argmax, width).float(), torch.floor(torch.div(argmax.float(), float(width)))
        argmax = torch.cat((argmax_x.view(batch_size, channels, -1), (argmax_y.view(batch_size, channels, -1))), 2)
        
        smax = self._activate_2d(x, self.temp)

        # compute x index (sum over y axis, produce with indices and then sum over x axis for the expectation)
        x_end_index = self.base_index + width * self.step_size
        x_indices = torch.arange(start=self.base_index, end=x_end_index, step=self.step_size).to(device)
        x_coords = torch.sum(torch.sum(smax, 2) * x_indices, 2)

        # compute y index (sum over x axis, produce with indices and then sum over y axis for the expectation)
        y_end_index = self.base_index + height * self.step_size
        y_indices = torch.arange(start=self.base_index, end=y_end_index, step=self.step_size).to(device)
        y_coords = torch.sum(torch.sum(smax, 3) * y_indices, 2)
        
        softargmax = torch.cat([torch.unsqueeze(x_coords, 2), torch.unsqueeze(y_coords, 2)], dim=2)
        # For debugging (testing if it's actually like the argmax?)
        # argmax_x = argmax_x.view(batch_size, channels)
        # argmax_y = argmax_y.view(batch_size, channels)
        # print("X err in soft argmax: {err}".format(err=torch.mean(torch.abs(argmax_x - x_coords))))
        # print("Y err in soft argmax: {err}".format(err=torch.mean(torch.abs(argmax_y - y_coords))))
        
        # Put the x coords and y coords (shape (B,C)) into an output with shape (B,C,2)
        return softargmax, argmax, smax
        
class SoftArgmax_PosEmb(nn.Module):
    def __init__(self,input_dim=16,num_point=16,normalize=True):
        super().__init__()
        self.init_conv = torch.nn.Conv2d(input_dim, num_point, 1, 1, 0)
        self.last_conv = torch.nn.Conv2d(num_point * 2, input_dim, 1, 1, 0)
        self.softargmax = SoftArgmax2D()
        self.normalize = normalize

    def forward(self,x):
        debug_info = {}
        B,C,H,W = x.shape
        device = x.device
        x = self.init_conv(x)
        uv_feature, _, smax = self.softargmax(x)
        
        if self.training == False:
            points = uv_feature.detach().cpu()
            smax = smax.detach().cpu()
            debug_info["atten_points"] = points
            debug_info["atten_mask"] = smax
        
        if self.normalize:
            uv_feature[:,:,0] = uv_feature[:,:,0] / W
            uv_feature[:,:,1] = uv_feature[:,:,1] / H

        uv_feature = rearrange(uv_feature, 'B C P -> B (P C)')
        uv_feature = repeat(uv_feature, 'B N -> B N H W',H=H,W=W)

        coordmap = self.create_coordmap(x, normalize=self.normalize, device=device)
        coordmap = coordmap - uv_feature
        feature = self.last_conv(coordmap)
        
        # return feature, points
        return feature, debug_info
    
    @staticmethod
    def create_coordmap(x, normalize=True):
        device = x.device
        B, C, H, W = x.shape
        xx_ones = torch.arange(W, dtype=torch.int32).to(device)
        xx_channel = repeat(xx_ones, 'W -> B C H W', B=B,C=C,H=H)

        yy_ones = torch.arange(H, dtype=torch.int32).to(device)
        yy_channel = repeat(yy_ones, 'H -> B C H W', B=B,C=C,W=W)

        xx_channel = xx_channel.float() 
        yy_channel = yy_channel.float()

        if normalize:
            xx_channel = xx_channel / (W - 1) * 2 - 1
            yy_channel = yy_channel / (H - 1) * 2 - 1

        pos_channel = torch.cat((xx_channel, yy_channel), 1)
        return pos_channel
    
    def get_feature(self, x, smax):
        B, C, H, W = x.shape
        _, P, _, _ = smax.shape
        x = repeat(x, 'B C H W -> B (C P) H W')
        smax = repeat(smax, 'B C H W -> B (C P) H W', P=P)
        

class SoftArgmax_Attention(nn.Module):
    def __init__(self,input_dim,num_point,heads=8,dim_head=0,softmax_temp=1.0):
        super().__init__()
        if dim_head == 0:
            dim_head = input_dim // heads
        
        self.emb_dim = heads * dim_head
        self.num_point = num_point
        self.heads = heads
        
        self.key_conv = nn.Conv2d(input_dim, self.emb_dim, 1, 1, 0)
        self.query_conv = nn.Conv2d(input_dim, self.emb_dim, 1, 1, 0)
        self.value_conv = nn.Conv2d(input_dim, self.emb_dim, 1, 1, 0)
        self.mask_conv = nn.Conv2d(input_dim, self.num_point, 1, 1, 0)
        
        self.softmax = torch.nn.Softmax(dim=3)
        self.softmax_temp = Parameter(torch.ones(1)*softmax_temp)
        
    def forward(self,x):
        debug_info = {}
        B,C,H,W = x.shape
        device = x.device
        query = self.query_conv(x) # B, C, H, W
        key = self.key_conv(x)
        value = self.value_conv(x) # B, C, H, W
        mask = self.mask_conv(x) # B, K, H, W
        
        query, key, value = map(lambda t: rearrange(t, 'B (c h) H W -> B h c H W', h = self.heads), (query, key, value))
        mask = self.softmax_2d(mask, self.softmax_temp) # B, K, H, W
        
        key_c = self.summarize(mask, key) # B h c K
        value_c = self.summarize(mask, value) # B h c K
        
        query = rearrange(query, 'B h c H W -> B h c (H W)')
        atten = torch.einsum('bhcn,bhck->bhnk',query,key_c)
        atten = self.softmax(atten) # B h (H W) K
        
        result = torch.einsum('bhnk,bhck->bhnc',atten,value_c)
        result = rearrange(result, 'B h (H W) c -> B (c h) H W', H=H, W=W)
        
        if self.training == False:
            debug_info["atten_points"] = self.get_points(mask)
            debug_info["atten_mask"] = mask.detach().cpu()
        return result, debug_info
    
    def summarize(self,mask,feature):
        B,h,c,H,W = feature.shape
        mask_c = repeat(mask, 'B K H W -> B h c K H W', h=h,c=c)
        feature_c = repeat(feature, 'B h c H W -> B h c K H W', K=self.num_point)
        feature_c = feature_c * mask_c
        feature_c = rearrange(feature_c, 'B h c K H W -> B h c K (H W)')
        feature_c = torch.sum(feature_c, 4)
        
        return feature_c
        
    def softmax_2d(self, x, temp):
        """
        For the lack of a true 2D softmax in pytorch, we reshape each image from (C, W, H) to (C, W*H) and then
        apply softmax, and then restore the original shape.
        :param x: A 4D tensor of shape (B, C, W, H) to apply softmax across the W and H dimensions
        :param temp: A scalar temperature to apply as part of the softmax function
        :return: Softmax(x, dims=(2,3))
        """
        B, C, W, H = x.size()
        x_flat = x.view((B, C, W*H)) / temp
        x_softmax = torch.nn.functional.softmax(x_flat, dim=2)
        return x_softmax.view((B, C, W, H))
    
    @staticmethod
    def create_coordmap(x):
        B, C, H, W = x.shape
        device = x.device
        xx_ones = torch.arange(W, dtype=torch.int32).to(device)
        xx_channel = repeat(xx_ones, 'W -> B C H W', B=B,C=C,H=H)

        yy_ones = torch.arange(H, dtype=torch.int32).to(device)
        yy_channel = repeat(yy_ones, 'H -> B C H W', B=B,C=C,W=W)

        xx_channel = xx_channel.float() 
        yy_channel = yy_channel.float()

        return xx_channel, yy_channel
    
    def get_points(self, mask):
        B, C, H, W = mask.shape
        x_channel, y_channel = self.create_coordmap(mask)
        x_value = x_channel * mask
        y_value = y_channel * mask
        
        x_value = rearrange(x_value, 'B C H W -> B C (H W)')
        y_value = rearrange(y_value, 'B C H W -> B C (H W)')
        
        x = torch.sum(x_value, 2, keepdim=True)
        y = torch.sum(y_value, 2, keepdim=True)
        
        return torch.cat([x, y], 2)