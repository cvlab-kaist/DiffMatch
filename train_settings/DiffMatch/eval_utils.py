import numpy as np
import torch as th
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image

from validation.metrics_uncertainty import compute_aucs


def correlation(src_feat, trg_feat, eps=1e-5):
    src_feat = src_feat / (src_feat.norm(dim=1, p=2, keepdim=True) + eps)
    trg_feat = trg_feat / (trg_feat.norm(dim=1, p=2, keepdim=True) + eps)

    return th.einsum('bchw, bcxy -> bhwxy', src_feat, trg_feat)

@th.no_grad()
def bilinear_sampler_sr(img, coords, mode='bilinear', mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""

    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = th.cat([xgrid, ygrid], dim=-1)
    r1, r2 = grid.shape[-3:-1]
    grid = rearrange(grid, '(b H W) r1 r2 C -> (b r1 r2) H W C', H=H, W=W)
    img = repeat(img, 'b C H W -> (b r1 r2) C H W', r1=r1, r2=r2)

    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return rearrange(img, '(b r1 r2) C H W -> b C H W (r1 r2)', r1=r1, r2=r2)


def resize_images_to_min_resolution(min_size, img, x, y, stride_net=16):  # for consistency with RANSAC-Flow
    """
    Function that resizes the image according to the minsize, at the same time resize the x,y coordinate.
    Extracted from RANSAC-Flow (https://github.com/XiSHEN0220/RANSAC-Flow/blob/master/evaluation/evalCorr/getResults.py)
    We here use exactly the same function that they used, for fair comparison. Even through the index_valid could
    also theoretically include the lower bound x = 0 or y = 0.
    """
    # Is is source image resized
    # Xs contains the keypoint x coordinate in source image
    # Ys contains the keypoints y coordinate in source image
    # valids is bool on wheter the keypoint is contained in the source image
    x = np.array(list(map(float, x.split(';')))).astype(np.float32)  # contains all the x coordinate
    y = np.array(list(map(float, y.split(';')))).astype(np.float32)

    w, h = img.size
    ratio = min(w / float(min_size), h / float(min_size))
    new_w, new_h = round(w / ratio), round(h / ratio)
    new_w, new_h = new_w // stride_net * stride_net, new_h // stride_net * stride_net

    ratioW, ratioH = new_w / float(w), new_h / float(h)
    img = img.resize((new_w, new_h), resample=Image.LANCZOS)

    x, y = x * ratioW, y * ratioH  # put coordinate in proper size after resizing the images
    index_valid = (x > 0) * (x < new_w) * (y > 0) * (y < new_h)

    return img, x, y, index_valid


def compute_pck_sparse_data(x_s, y_s, x_r, y_r, flow, pck_thresholds, dict_list_uncertainties, uncertainty_est=None):
    flow_x = flow[0, 0].cpu().numpy()
    flow_y = flow[0, 1].cpu().numpy()

    # remove points for which xB, yB are outside of the image
    h, w = flow_x.shape
    index_valid = (
        (np.int32(np.round(x_r)) >= 0)
        * (np.int32(np.round(x_r)) < w)
        * (np.int32(np.round(y_r)) >= 0)
        * (np.int32(np.round(y_r)) < h)
    )
    x_s, y_s, x_r, y_r = x_s[index_valid], y_s[index_valid], x_r[index_valid], y_r[index_valid]
    nbr_valid_corr = index_valid.sum()

    # calculates the PCK
    if nbr_valid_corr > 0:
        # more accurate to compute the flow like this, instead of rounding both coordinates as in RANSAC-Flow
        flow_gt_x = x_s - x_r
        flow_gt_y = y_s - y_r
        flow_est_x = flow_x[np.int32(np.round(y_r)), np.int32(np.round(x_r))]
        flow_est_y = flow_y[np.int32(np.round(y_r)), np.int32(np.round(x_r))]
        EPE = ((flow_gt_x - flow_est_x) ** 2 + (flow_gt_y - flow_est_y) ** 2) ** 0.5
        EPE = EPE.reshape((-1, 1))
        AEPE = np.mean(EPE)
        count_pck = np.sum(EPE <= pck_thresholds, axis=0)
        # here compares the EPE of the pixels to be inferior to some value pixelGrid
    else:
        count_pck = np.zeros(pck_thresholds.shape[1])
        AEPE = np.nan

    results = {'count_pck': count_pck, 'nbr_valid_corr': nbr_valid_corr, 'aepe': AEPE}

    # calculates sparsification plot information
    if uncertainty_est is not None:
        flow_est = th.from_numpy(np.concatenate((flow_est_x.reshape(-1, 1), flow_est_y.reshape(-1, 1)), axis=1))
        flow_gt = th.from_numpy(np.concatenate((flow_gt_x.reshape(-1, 1), flow_gt_y.reshape(-1, 1)), axis=1))

        # uncert shape is #number_of_elements
        for uncertainty_name in uncertainty_est.keys():
            if (
                uncertainty_name == 'inference_parameters'
                or uncertainty_name == 'log_var_map'
                or uncertainty_name == 'weight_map'
                or uncertainty_name == 'warping_mask'
            ):
                continue

            if 'p_r' == uncertainty_name:
                # convert confidence map to uncertainty
                uncert = (1.0 / (uncertainty_est['p_r'] + 1e-6)).squeeze()[
                    np.int32(np.round(y_r)), np.int32(np.round(x_r))
                ]
            else:
                uncert = uncertainty_est[uncertainty_name].squeeze()[np.int32(np.round(y_r)), np.int32(np.round(x_r))]
            # compute metrics based on uncertainty
            uncertainty_metric_dict = compute_aucs(flow_gt, flow_est, uncert, intervals=50)
            if uncertainty_name not in dict_list_uncertainties.keys():
                # for first image, create the list for each possible uncertainty type
                dict_list_uncertainties[uncertainty_name] = []
            dict_list_uncertainties[uncertainty_name].append(uncertainty_metric_dict)
    return results, dict_list_uncertainties


@th.no_grad()
def extract_features_sr(pyramid, im_target, im_source):
    im_target_pyr = pyramid(im_target, eigth_resolution=False)
    im_source_pyr = pyramid(im_source, eigth_resolution=False)

    C11 = im_target_pyr[0]
    C21 = im_source_pyr[0]
    C12 = im_target_pyr[1]
    C22 = im_source_pyr[1]
    C13 = im_target_pyr[2]
    C23 = im_source_pyr[2]

    return C11, C21, C12, C22, C13, C23 


@th.no_grad()
def extract_raw_features_sr(pyramid, target, source, feature_size=512):
    C11, C21, C12, C22, C13, C23 = extract_features_sr(pyramid, target, source)
    trg_feat = [C12]
    src_feat = [C22]
    trg_feat_list = []
    src_feat_list = []

    for trg, src in zip(trg_feat, src_feat):
        trg_feat_list.append(F.interpolate(trg, size=(feature_size), mode='bilinear', align_corners=True))
        src_feat_list.append(F.interpolate(src, size=(feature_size), mode='bilinear', align_corners=True))

    trg_feat = F.interpolate(C11, size=feature_size, mode='bilinear', align_corners=False)

    return trg_feat_list, src_feat_list, trg_feat


@th.no_grad()
def local_feat_sr(src_feats_list, coords, radius):
    src_feat_sampled_list = []
    for src_feat in src_feats_list:
        coords_ = coords
        r = radius
        coords_ = coords_.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords_.shape

        dx = th.linspace(-r, r, 2 * r + 1, device=coords_.device)
        dy = th.linspace(-r, r, 2 * r + 1, device=coords_.device)
        delta = th.stack(th.meshgrid(dy, dx), axis=-1)

        centroid_lvl = coords_.reshape(batch * h1 * w1, 1, 1, 2)
        delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
        coords_lvl = centroid_lvl + delta_lvl

        src_feat_sampled = bilinear_sampler_sr(src_feat, coords_lvl)
        src_feat_sampled_list.append(src_feat_sampled)

    return src_feat_sampled_list


@th.no_grad()
def build_local_corr_sr(src_feat_sampled_list, trg_feat_list):
    corr_list = []
    for src_feat, trg_feat in zip(src_feat_sampled_list, trg_feat_list):
        eps = 1e-5
        src_feat = src_feat / (src_feat.norm(dim=1, p=2, keepdim=True) + eps)
        trg_feat = trg_feat / (trg_feat.norm(dim=1, p=2, keepdim=True) + eps)

        corr = th.einsum('bchwn,bchw->bhwn', src_feat, trg_feat)

    corr_list.append(corr)
    raw_corr = sum(corr_list) / len(corr_list)

    return raw_corr


@th.no_grad()
def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""

    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = th.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


@th.no_grad()
def coords_grid(batch, ht, wd, device):
    coords = th.meshgrid(th.arange(ht, device=device), th.arange(wd, device=device))
    coords = th.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


@th.no_grad()
def initialize_flow(B, H, W, device):
    """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""

    coords = coords_grid(B, H, W, device=device)

    return coords


@th.no_grad()
def local_Corr(corr, coords, radius):
    corr = rearrange(corr, 'b c hs ws ht wt -> (b ht wt) c hs ws')
    r = radius
    coords = coords.permute(0, 2, 3, 1)
    batch, h1, w1, _ = coords.shape

    dx = th.linspace(-r, r, 2 * r + 1, device=coords.device)
    dy = th.linspace(-r, r, 2 * r + 1, device=coords.device)
    delta = th.stack(th.meshgrid(dy, dx), axis=-1)

    centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2)
    delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
    coords_lvl = centroid_lvl + delta_lvl

    corr = bilinear_sampler(corr, coords_lvl)
    corr = corr.view(batch, h1, w1, -1)

    return corr.permute(0, 3, 1, 2).contiguous().float()


@th.no_grad()
def chunk_cosine_sim(x, y):
    # Normalize x and y along the last dimension
    x_normalized = F.normalize(x, p=2, dim=-1)
    y_normalized = F.normalize(y, p=2, dim=-1)

    # Transpose y to align dimensions for multiplication
    y_transposed = y_normalized.permute(0, 1, 3, 2)

    # Compute cosine similarity using matrix multiplication
    cosine_similarity = th.matmul(x_normalized, y_transposed)

    return cosine_similarity  # Bx1x(t_x)x(t_y)

def extract_features(
    pyramid,
    im_target,
    im_source,
    im_target_256,
    im_source_256,
    im_target_pyr=None,
    im_source_pyr=None,
    im_target_pyr_256=None,
    im_source_pyr_256=None,
):
    if im_target_pyr is None:
        im_target_pyr = pyramid(im_target, eigth_resolution=True)
    if im_source_pyr is None:
        im_source_pyr = pyramid(im_source, eigth_resolution=True)
    c10 = im_target_pyr[-3]
    c20 = im_target_pyr[-3]
    c11 = im_target_pyr[-2]  # load_size original_res/4xoriginal_res/4
    c21 = im_source_pyr[-2]
    c12 = im_target_pyr[-1]  # load_size original_res/8xoriginal_res/8
    c22 = im_source_pyr[-1]

    # pyramid, 256 reso
    if im_target_pyr_256 is None:
        im_target_pyr_256 = pyramid(im_target_256)
    if im_source_pyr_256 is None:
        im_source_pyr_256 = pyramid(im_source_256)
    c13 = im_target_pyr_256[-2]  # load_size 256/8 x 256/8
    c23 = im_source_pyr_256[-2]  # load_size 256/8 x 256/8
    c14 = im_target_pyr_256[-1]  # load_size 256/16 x 256/16
    c24 = im_source_pyr_256[-1]  # load_size 256/16 x 256/16

    return c14, c24, c13, c23, c12, c22, c11, c21, c10, c20

@th.no_grad()
def extract_raw_features(pyramid, target, source, target_256, source_256, feature_size=64):
    c14, c24, c13, c23, c12, c22, c11, c21, c10, c20 = extract_features(
        pyramid, target, source, target_256, source_256, None, None, None, None
    )
    trg_feat = [c14, c13, c12, c11]
    src_feat = [c24, c23, c22, c21]
    trg_feat_list = []
    src_feat_list = []
    corr_list = []
    for trg, src in zip(trg_feat, src_feat):
        trg_feat_list.append(F.interpolate(trg, size=(feature_size), mode='bilinear', align_corners=True))
        src_feat_list.append(F.interpolate(src, size=(feature_size), mode='bilinear', align_corners=True))

    for trg, src in zip(trg_feat_list, src_feat_list):
        corr = correlation(src, trg)[:, None]
        corr_list.append(corr)

    raw_corr = sum(corr_list) / len(corr_list)

    c10 = F.interpolate(c10, size=(feature_size), mode='bilinear', align_corners=True)
    c20 = F.interpolate(c22, size=(feature_size), mode='bilinear', align_corners=True)
    return raw_corr, c10, c20

@th.no_grad()
def extract_resnet_features(resnet, target, source, feature_size):
    """Extract and build correlations using ResNet intermediate features

    Args:
        resnet (nn.Module): ResNet backbone
        target (torch.Tensor): [B, 3, 512, 512] Target image
        source (torch.Tensor): [B, 3, 512, 512] Source image
        feature_size (int): Feature size to extract, default 64
    """
    # Extract features from ResNet

    trg_feat = resnet(target)
    src_feat = resnet(source)
    trg_feat_list = []
    src_feat_list = []
    corr_list = []
    for trg, src in zip(trg_feat, src_feat):
        trg_feat_list.append(F.interpolate(trg, size=(feature_size), mode='bilinear', align_corners=True))
        src_feat_list.append(F.interpolate(src, size=(feature_size), mode='bilinear', align_corners=True))
    for trg, src in zip(trg_feat_list, src_feat_list):
        corr = correlation(src, trg)[:, None]
        corr_list.append(corr)
    raw_corr = sum(corr_list) / len(corr_list)

    return raw_corr, None, None