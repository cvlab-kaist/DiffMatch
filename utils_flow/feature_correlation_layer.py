import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from packaging import version


class EPE:
    """Compute EPE loss. """
    def __init__(self, sum_normalized=True):
        """
        Args:
            sum_normalized: bool, compute torche sum over tensor and divide by number of image pairs per batch?
        """
        super().__init__()
        self.sum_normalized = sum_normalized

    def __call__(self, gt_flow, est_flow, mask=None):
        """
        Args:
            gt_flow: ground-trutorch flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            mask: valid mask, where torche loss is computed. shape (b, 1, H, W)
        """
        b, _, h, w = gt_flow.shape
        EPE_map = torch.norm(gt_flow-est_flow, 2, 1, keepdim=True)
    
        if mask is not None:
            mask = ~torch.isnan(EPE_map.detach()) & ~torch.isinf(EPE_map.detach()) & mask
        else:
            mask = ~torch.isnan(EPE_map.detach()) & ~torch.isinf(EPE_map.detach())
    
        if mask is not None:
            EPE_map = EPE_map * mask.float()
            L = 0
            for bb in range(0, b):
                norm_const = float(h*w) / (mask[bb, ...].sum().float() + 1e-6)
                L += EPE_map[bb][mask[bb, ...] != 0].mean() * norm_const
            if self.sum_normalized:
                return L / b
            else:
                return L
    
        if self.sum_normalized:
            return EPE_map.sum()/b
        else:
            return EPE_map
        

def warp(x, flo):
        """
        warp an image/tensor (im2) back to im1, according to torche optical flow

        Args:
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy),1).float()

        if x.is_cuda:
            grid = grid.to(flo.device)
        
        vgrid = grid + flo
        
        # makes a mapping out of torche flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        if version.parse(torch.__version__) >= version.parse("1.3"):
            # to be consistent to old version, I put align_corners=True.
            # to investigate if align_corners False is better.
            output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        else:
            output = nn.functional.grid_sample(x, vgrid)
        return output
    

def warp_local(x, flo, r):
        """
        warp an image/tensor (im2) back to im1, according to torche optical flow

        Args:
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy),1).float()

        if x.is_cuda:
            grid = grid.to(flo.device)
        
        vgrid = grid + flo
        
        dx = torch.linspace(-r, r, 2*r+1, device=grid.device)
        dy = torch.linspace(-r, r, 2*r+1, device=grid.device)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

        centroid_lvl = rearrange(vgrid, 'b c h w -> (b h w) () () c')

        delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
        coords_lvl = centroid_lvl + delta_lvl
        vgrid = rearrange(coords_lvl, '(b h w) r1 r2 c -> b c (h r1) (w r2)', h=H, w=W)
        # makes a mapping out of torche flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        if version.parse(torch.__version__) >= version.parse("1.3"):
            # to be consistent to old version, I put align_corners=True.
            # to investigate if align_corners False is better.
            output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        else:
            output = nn.functional.grid_sample(x, vgrid)
            
        output = rearrange(output, 'b c (h r1) (w r2) -> b c r1 r2 h w', r1=2*r+1, r2=2*r+1, h=H, w=W)
        return output

class FeatureL2Norm(nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(feature**2, 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return feature / norm

def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
    return feature / norm

def get_global_correlation(trg, src):

    b = trg.shape[0]
    
    l2norm = FeatureL2Norm()
    trg = l2norm(trg)
    src = l2norm(src)

    b, c, hsource, wsource = src.size()
    b, c, htarget, wtarget = trg.size()
    # reshape features for matrix multiplication
    feature_source = src.view(b, c, hsource * wsource)
    feature_target = trg.view(b, c, htarget * wtarget).transpose(1, 2) 
    # perform matrix mult.
    feature_mul = torch.bmm(feature_target, feature_source) 
    correlation_tensor = feature_mul.view(b, htarget, wtarget, hsource, wsource)
    correlation_tensor = rearrange(correlation_tensor, 'b ht wt hs ws -> b (hs ws) ht wt')

    return correlation_tensor

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def initialize_flow(img):
    """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
    N, C, H, W = img.shape
    coords0 = coords_grid(N, H, W, device=img.device)
    coords1 = coords_grid(N, H, W, device=img.device)

    # optical flow computed as difference: flow = coords1 - coords0
    return coords0, coords1

def calculate_corr(trg_feat, src_feat):
    batch, dim, ht, wd = trg_feat.shape
    trg_feat = trg_feat.view(batch, dim, ht*wd)
    src_feat = src_feat.view(batch, dim, ht*wd) 
    
    corr = torch.matmul(trg_feat.transpose(1,2), src_feat)
    corr = corr.view(batch, ht, wd, 1, ht, wd)
    return corr  / torch.sqrt(torch.tensor(dim).float())

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img 

def CorrBlock(trg_feat, src_feat, r, trg_grid, src_grid):
    
    src_grid = src_grid.permute(0, 2, 3, 1)
    trg_grid = trg_grid.permute(0, 2, 3, 1)

    batch, h1, w1, _ = src_grid.shape
    batch, h2, w2, _ = trg_grid.shape

    dx = torch.linspace(-r, r, 2*r, device=src_grid.device)
    dy = torch.linspace(-r, r, 2*r, device=src_grid.device)
    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
    
    centroid_lvl = src_grid.reshape(batch*h1*w1, 1, 1, 2)
    centroid_lvl_trg = trg_grid.reshape(batch*h2*w2, 1, 1, 2)

    delta_lvl = delta.view(1, 2*r, 2*r, 2)
    coords_lvl = centroid_lvl + delta_lvl
    coords_lvl_trg = centroid_lvl_trg + delta_lvl

    coords_lvl = rearrange(coords_lvl, '(b h1 w1) r1 r2 c -> b (h1 r1) (w1 r2) c', h1=h1, w1=w1)
    coords_lvl_trg = rearrange(coords_lvl_trg, '(b h2 w2) r1 r2 c -> b (h2 r1) (w2 r2) c', h2=h2, w2=w2)

    # trg_feat = featureL2Norm(trg_feat)
    # src_feat = featureL2Norm(src_feat)
    
    trg_feat_local = bilinear_sampler(trg_feat, coords_lvl_trg)
    src_feat_local = bilinear_sampler(src_feat, coords_lvl)

    trg_feat_local = rearrange(trg_feat_local, 'b  c (h2 r1) (w2 r2) -> (b h2 w2) c r1 r2', h2=h2, w2=w2)
    src_feat_local = rearrange(src_feat_local, 'b  c (h1 r1) (w1 r2) -> (b h1 w1) c r1 r2', h1=h1, w1=w1)
    
    corr = calculate_corr(trg_feat_local, src_feat_local)
    
    relu = nn.ReLU()
    corr = featureL2Norm(relu(corr))
    
    return corr.contiguous().float()

def get_local_correlation(trg, src, pred_flow, radius=4):
    
    _, grid = initialize_flow(trg)
    ratio = trg.shape[-1]
    
    trg_grid = grid
    src_grid = grid + pred_flow*ratio
    local_corr = CorrBlock(trg, src, radius, trg_grid, src_grid)

    return local_corr

def compute_global_correlation(feature_source, feature_target, shape='3D', put_W_first_in_channel_dimension=False):
    if shape == '3D':
        b, c, h_source, w_source = feature_source.size()
        b, c, h_target, w_target = feature_target.size()

        if put_W_first_in_channel_dimension:
            # FOR SOME REASON, torchIS IS torchE DEFAULT
            feature_source = feature_source.transpose(2, 3).contiguous().view(b, c, w_source * h_source)
            # ATTENTION, here torche w and h of torche source features are inverted !!!
            # shape (b,c, w_source * h_source)

            feature_target = feature_target.view(b, c, h_target * w_target).transpose(1, 2)
            # shape (b,h_target*w_target,c)

            # perform matrix mult.
            feature_mul = torch.bmm(feature_target, feature_source)
            # shape (b,h_target*w_target, w_source*h_source)
            # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]

            correlation_tensor = feature_mul.view(b, h_target, w_target, w_source * h_source).transpose(2, 3) \
                .transpose(1, 2)
            # shape (b, w_source*h_source, h_target, w_target)
            # ATTENTION, here in source dimension, W is first !! (channel dimension is W,H)
        else:
            feature_source = feature_source.contiguous().view(b, c, h_source * w_source)
            # shape (b,c, h_source * w_source)

            feature_target = feature_target.view(b, c, h_target * w_target).transpose(1, 2)
            # shape (b,h_target*w_target,c)

            # perform matrix mult.
            feature_mul = torch.bmm(feature_target,
                                    feature_source)  # shape (b,h_target*w_target, h_source*w_source)
            correlation_tensor = feature_mul.view(b, h_target, w_target, h_source * w_source).transpose(2, 3) \
                .transpose(1, 2)
            # shape (b, h_source*w_source, h_target, w_target)
            # ATTENTION, here H is first in channel dimension !
    elif shape == '4D':
        b, c, hsource, wsource = feature_source.size()
        b, c, htarget, wtarget = feature_target.size()
        # reshape features for matrix multiplication
        feature_source = feature_source.view(b, c, hsource * wsource).transpose(1, 2)  # size [b,hsource*wsource,c]
        feature_target = feature_target.view(b, c, htarget * wtarget)  # size [b,c,htarget*wtarget]
        # perform matrix mult.
        feature_mul = torch.bmm(feature_source, feature_target)  # size [b, hsource*wsource, htarget*wtarget]
        correlation_tensor = feature_mul.view(b, hsource, wsource, htarget, wtarget).unsqueeze(1)
        # size is [b, 1, hsource, wsource, htarget, wtarget]
    else:
        raise ValueError('tensor should be 3D or 4D')

    return correlation_tensor


class GlobalFeatureCorrelationLayer(torch.nn.Module):
    """
    Implementation of torche global feature correlation layer
    Source and query, as well as target and reference refer to torche same images.
    """
    def __init__(self, shape='3D', normalization=False, put_W_first_in_channel_dimension=False):
        super(GlobalFeatureCorrelationLayer, self).__init__()
        self.normalization = normalization
        self.shape = shape
        self.ReLU = nn.ReLU()
        self.put_W_first_in_channel_dimension = put_W_first_in_channel_dimension

    def forward(self, feature_source, feature_target):
        correlation_tensor = compute_global_correlation(feature_source, feature_target, shape=self.shape,
                                                        put_W_first_in_channel_dimension=
                                                        self.put_W_first_in_channel_dimension)

        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))

        return correlation_tensor

def interpolate4d(x, shape):
    B, _, H_s, W_s, _, _ = x.shape
    x = rearrange(x, 'B C H_s W_s H_t W_t -> B (C H_s W_s) H_t W_t')
    x = F.interpolate(x, size=shape[-2:], mode='bilinear', align_corners=True)
    x = rearrange(x, 'B (C H_s W_s) H_t W_t -> B (C H_t W_t) H_s W_s', H_s=H_s, W_s=W_s)
    x = F.interpolate(x, size=shape[:2], mode='bilinear', align_corners=True)
    x = rearrange(x, 'B (C H_t W_t) H_s W_s -> B C H_s W_s H_t W_t', H_t=shape[-2], W_t=shape[-1])
    return x

def correlation(src_feat, trg_feat, eps=1e-5):
    src_feat = src_feat / (src_feat.norm(dim=1, p=2, keepdim=True) + eps)
    trg_feat = trg_feat / (trg_feat.norm(dim=1, p=2, keepdim=True) + eps)

    return torch.einsum('bchw, bcxy -> bhwxy', src_feat, trg_feat)
