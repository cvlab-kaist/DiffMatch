import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from PIL import Image
from torchvision.utils import save_image

from datasets.utils import flow_viz
from utils_flow.feature_correlation_layer import warp


def visualize(sample, category, rate, name_dataset, i, batch_vis, source_vis, target_vis, mask):
    os.makedirs(f'vis_{category}/{rate}_{name_dataset}/image_samples', exist_ok=True)
    os.makedirs(f'vis_{category}/{rate}_{name_dataset}/gt_samples', exist_ok=True)
    os.makedirs(f'vis_{category}/{rate}_{name_dataset}/src_samples', exist_ok=True)
    os.makedirs(f'vis_{category}/{rate}_{name_dataset}/trg_samples', exist_ok=True)
    os.makedirs(f'vis_{category}/{rate}_{name_dataset}/warped_samples', exist_ok=True)
    os.makedirs(f'vis_{category}/{rate}_{name_dataset}/warped_gt_samples', exist_ok=True)
    os.makedirs(f'vis_{category}/{rate}_{name_dataset}/mask', exist_ok=True)
    
    for j in range(len(sample)):
        flow_vis = sample[j].detach().permute(1,2,0).float().cpu().numpy()
        flow_vis = flow_viz.flow_to_image(flow_vis)
        plt.imsave(f'vis_{category}/{rate}_{name_dataset}/image_samples/flow_{i}_{j}.png', flow_vis / 255.0)

        flow_gt_vis = batch_vis[j].detach().permute(1,2,0).float().cpu().numpy()
        flow_gt_vis = flow_viz.flow_to_image(flow_gt_vis)
        plt.imsave(f'vis_{category}/{rate}_{name_dataset}/gt_samples/gt_{i}_{j}.png', flow_gt_vis / 255.0)

        src = source_vis[j].permute(1, 2, 0).cpu().numpy()
        src = Image.fromarray((src).astype(np.uint8))
        src.save(f'vis_{category}/{rate}_{name_dataset}/src_samples/src_{i}_{j}.png')

        trg = target_vis[j].permute(1, 2, 0).cpu().numpy()
        trg = Image.fromarray((trg).astype(np.uint8))
        trg.save(f'vis_{category}/{rate}_{name_dataset}/trg_samples/trg_{i}_{j}.png')
        
        warped_src = warp(source_vis[j:j+1].to(sample.device).float(), sample)
        warped_src_masked = warped_src * mask[j:j+1].float()
        warped_src = warped_src[j].permute(1, 2, 0).detach().cpu().numpy()
        warped_src_masked = warped_src_masked[j].permute(1, 2, 0).detach().cpu().numpy()
        warped_src = Image.fromarray((warped_src).astype(np.uint8))
        warped_src_masked = Image.fromarray((warped_src_masked).astype(np.uint8))
        warped_src.save(f'vis_{category}/{rate}_{name_dataset}/warped_samples/warped_{i}_{j}.png')
        warped_src_masked.save(f'vis_{category}/{rate}_{name_dataset}/warped_samples/warped_masked_{i}_{j}.png')
        
        warped_gt = warp(source_vis[j:j+1].to(batch_vis.device).float(), batch_vis)
        warped_gt_masked = warped_gt * mask[j:j+1].float()
        warped_gt = warped_gt[j].permute(1, 2, 0).detach().cpu().numpy()
        warped_gt_masked = warped_gt_masked[j].permute(1, 2, 0).detach().cpu().numpy()
        warped_gt = Image.fromarray((warped_gt).astype(np.uint8))
        warped_gt_masked = Image.fromarray((warped_gt_masked).astype(np.uint8))
        warped_gt.save(f'vis_{category}/{rate}_{name_dataset}/warped_gt_samples/warped_{i}_{j}.png')
        warped_gt_masked.save(f'vis_{category}/{rate}_{name_dataset}/warped_gt_samples/warped_masked_{i}_{j}.png')

        mask_vis = th.stack((mask[j], mask[j], mask[j]))
        save_image(mask_vis.float(), f'vis_{category}/{rate}_{name_dataset}/mask/mask_{i}_{j}.png')
    