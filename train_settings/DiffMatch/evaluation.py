import os

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.geometric_matching_datasets.ETH3D_interval import ETHInterval
from train_settings.DiffMatch.feature_backbones.VGG_features import VGGPyramid
from training.actors.batch_processing import GLUNetBatchPreprocessing
from utils_flow.correlation_to_matches_utils import \
    correlation_to_flow_w_argmax
from utils_flow.visualization_utils import visualize

from .eval_utils import (build_local_corr_sr, extract_raw_features,
                         extract_raw_features_sr, initialize_flow, local_Corr,
                         local_feat_sr)
from .improved_diffusion import dist_util
from .improved_diffusion.gaussian_diffusion import GaussianDiffusion


def prepare_data(settings, batch_preprocessing, SIZE, data):
    source_vis = data['source_image']  # B, C, 512, 512
    target_vis = data['target_image']
    _, _, H_ori, W_ori = source_vis.shape

    data = batch_preprocessing(data)

    source = data['source_image']  # B, C, 512, 512
    target = data['target_image']
    batch_ori = data['flow_map']
    target = F.interpolate(target, size=512, mode='bilinear', align_corners=False)
    source = F.interpolate(source, size=512, mode='bilinear', align_corners=False)
    source_256 = data['source_image_256'].to(dist_util.dev())
    target_256 = data['target_image_256'].to(dist_util.dev())

    if settings.env.eval_dataset == 'hp-240':
        source_256 = source
        target_256 = target

    else:
        source_256 = data['source_image_256']
        target_256 = data['target_image_256']
    mask = data['correspondence_mask']
    return data, H_ori, W_ori, source, target, batch_ori, source_256, target_256, source_vis, target_vis, mask

def run_sample_sr(settings, logger, diffusion_sr, model_sr, pyramid, data, sample, sample_sr):
    feature_size = 256
    image_size = 256
    radius = 4

    source = data['source_image'].to(dist_util.dev())  # B, C, 512, 512
    target = data['target_image'].to(dist_util.dev())
    batch_ori = data['flow_map'].to(dist_util.dev())

    target = F.interpolate(target, size=512, mode='bilinear', align_corners=False)
    source = F.interpolate(source, size=512, mode='bilinear', align_corners=False)

    with th.no_grad():
        trg_feat_list, src_feat_list, trg_feat = extract_raw_features_sr(pyramid, target, source, feature_size)
       
    if not sample_sr == None:
        init_flow_sr = sample_sr
    else:
        init_flow_sr = F.interpolate(sample, size=feature_size, mode='bilinear', align_corners=True)

    init_flow_sr[:, 0] *= feature_size
    init_flow_sr[:, 1] *= feature_size

    coords_sr = initialize_flow(init_flow_sr.shape[0], feature_size, feature_size, dist_util.dev())
    coords_warped_sr = coords_sr + init_flow_sr

    sampled_src_feat_list = local_feat_sr(src_feat_list, coords_warped_sr.to(dist_util.dev()), radius)
    local_corr_sr = build_local_corr_sr(sampled_src_feat_list, trg_feat_list).permute(0, 3, 1, 2)

    init_flow_sr /= feature_size

    model_kwsettings_sr = {'y': None, 'local_corr': local_corr_sr, 'init_flow': init_flow_sr}

    logger.info(f"Starting SR sampling")

    sample_sr, prev_var = diffusion_sr.ddim_sample_loop(
        model_sr,
        (1, 2, image_size, image_size),
        noise=None,
        clip_denoised=settings.env.clip_denoised,
        model_kwargs=model_kwsettings_sr,
        eta=0.0,
        progress=True,
        denoised_fn=None,
        sampling_kwargs={'src_img': source, 'trg_img': target},
        logger=logger,
        n_batch=settings.env.n_batch
    )
    sample_sr = th.clamp(sample_sr, min=-1, max=1)
    return batch_ori, sample_sr, init_flow_sr


def run_sample_lr(
    settings, logger, diffusion, model, radius, source, target, feature_size, raw_corr, trg_feat, init_flow
):
    init_flow = init_flow * feature_size
    coords = initialize_flow(init_flow.shape[0], feature_size, feature_size, dist_util.dev())
    coords_warped = coords + init_flow

    local_corr = local_Corr(
        raw_corr.view(1, 1, feature_size, feature_size, feature_size, feature_size).to(dist_util.dev()),
        coords_warped.to(dist_util.dev()),
        radius,
    )

    local_corr = F.interpolate(
        local_corr.view(1, (2 * radius + 1) ** 2, feature_size, feature_size),
        size=feature_size,
        mode='bilinear',
        align_corners=True,
    )

    init_flow = F.interpolate(init_flow, size=feature_size, mode='bilinear', align_corners=True)
    init_flow /= feature_size

    model_kwsettings = {'y': None, 'local_corr': local_corr, 'init_flow': init_flow, 'trg_feat': trg_feat}

    image_size_h, image_size_w = feature_size, feature_size

    logger.info(f"\nStarting sampling")

    sample, _ = diffusion.ddim_sample_loop(
        model,
        (1, 2, image_size_h, image_size_w),
        noise=None,
        clip_denoised=settings.env.clip_denoised,
        model_kwargs=model_kwsettings,
        eta=0.0,
        progress=True,
        denoised_fn=None,
        sampling_kwargs={'src_img': source, 'trg_img': target},
        logger=logger,
        n_batch=settings.env.n_batch,
    )

    sample = th.clamp(sample, min=-1, max=1)
    return sample

def run_evaluation_eth3d(
    settings,
    data_dir,
    input_images_transform,
    gt_flow_transform,
    co_transform,
    logger,
    diffusion,
    model,
    diffusion_sr,
    model_sr,
    severity,
    corruption_number,
):
    # ETH3D dataset information
    dataset_names = [
        'lakeside',
        'sand_box',
        'storage_room',
        'storage_room_2',
        'tunnel',
        'delivery_area',
        'electro',
        'forest',
        'playground',
        'terrains',
    ]
    rates = [3, 5, 7, 9, 11, 13, 15]
    dict_results = {}
    for rate in rates:
        logger.info('Computing results for interval {}...'.format(rate))
        dict_results['rate_{}'.format(rate)] = {}
        list_of_outputs_per_rate = []
        num_pck_1 = 0.0
        num_pck_3 = 0.0
        num_pck_5 = 0.0
        num_valid_correspondences = 0.0
        for name_dataset in dataset_names:
            logger.info('looking at dataset {}...'.format(name_dataset))
            test_set = ETHInterval(
                root=data_dir,
                path_list=os.path.join(
                    data_dir, 'info_ETH3D_files', '{}_every_5_rate_of_{}'.format(name_dataset, rate)
                ),
                source_image_transform=input_images_transform,
                target_image_transform=input_images_transform,
                flow_transform=gt_flow_transform,
                co_transform=co_transform,
                severity=severity,
                corruption_number=corruption_number,
            )  # only test
            test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)
            logger.info(f"Length of test set: {test_set.__len__()}")

            (_, _, _, _, _), output = run_evaluation_generic(
                settings, logger, test_dataloader, diffusion, model, diffusion_sr, model_sr, rate, name_dataset
            )
            # to save the intermediate results
            # dict_results['rate_{}'.format(rate)][name_dataset] = output
            list_of_outputs_per_rate.append(output)
            num_pck_1 += output['num_pixels_pck_1']
            num_pck_3 += output['num_pixels_pck_3']
            num_pck_5 += output['num_pixels_pck_5']
            num_valid_correspondences += output['num_valid_corr']

        # average over all datasets for this particular rate of interval
        avg = {
            'AEPE': np.mean([list_of_outputs_per_rate[i]['AEPE'] for i in range(len(dataset_names))]),
            'AEPE_init': np.mean([list_of_outputs_per_rate[i]['AEPE_init'] for i in range(len(dataset_names))]),
            'AEPE_VGG64': np.mean([list_of_outputs_per_rate[i]['AEPE_VGG64'] for i in range(len(dataset_names))]),
            'PCK_1_per_image': np.mean(
                [list_of_outputs_per_rate[i]['PCK_1_per_image'] for i in range(len(dataset_names))]
            ),
            'PCK_3_per_image': np.mean(
                [list_of_outputs_per_rate[i]['PCK_3_per_image'] for i in range(len(dataset_names))]
            ),
            'PCK_5_per_image': np.mean(
                [list_of_outputs_per_rate[i]['PCK_5_per_image'] for i in range(len(dataset_names))]
            ),
            'pck-1-per-rate': num_pck_1 / (num_valid_correspondences + 1e-6),
            'pck-3-per-rate': num_pck_3 / (num_valid_correspondences + 1e-6),
            'pck-5-per-rate': num_pck_5 / (num_valid_correspondences + 1e-6),
            'num_valid_corr': num_valid_correspondences,
        }
        logger.info('Results for rate {}:'.format(rate))
        logger.info(avg)
        dict_results['rate_{}'.format(rate)] = avg

    avg_rates = {
        'AEPE': np.mean([dict_results['rate_{}'.format(rate)]['AEPE'] for rate in rates]),
        'PCK_1_per_image': np.mean([dict_results['rate_{}'.format(rate)]['PCK_1_per_image'] for rate in rates]),
        'PCK_3_per_image': np.mean([dict_results['rate_{}'.format(rate)]['PCK_3_per_image'] for rate in rates]),
        'PCK_5_per_image': np.mean([dict_results['rate_{}'.format(rate)]['PCK_5_per_image'] for rate in rates]),
        'pck-1-per-rate': np.mean([dict_results['rate_{}'.format(rate)]['pck-1-per-rate'] for rate in rates]),
        'pck-3-per-rate': np.mean([dict_results['rate_{}'.format(rate)]['pck-3-per-rate'] for rate in rates]),
        'pck-5-per-rate': np.mean([dict_results['rate_{}'.format(rate)]['pck-5-per-rate'] for rate in rates]),
    }
    dict_results['avg'] = avg_rates
    return dict_results

def run_evaluation_generic(
    settings, logger, val_loader, diffusion: GaussianDiffusion, model, diffusion_sr, model_sr, rate, name_dataset
):
    keys = ['image_samples', 'gt_samples', 'src_samples', 'trg_samples', 'mask', 'warped_samples', 'warped_gt_samples']
    category = 'hp' if 'hp' in settings.env.eval_dataset else 'eth3d' if 'eth3d' in settings.env.eval_dataset else None
    for key in keys:
        os.makedirs(f'vis_{category}/{rate}_{name_dataset}/{key}', exist_ok=True)

    batch_preprocessing = GLUNetBatchPreprocessing(
        settings, apply_mask=False, apply_mask_zero_borders=False, sparse_ground_truth=False
    )

    logger.log("sampling...")

    mean_epe_list, mean_epe_init_list, epe_all_list, epe_init_all_list = [], [], [], []
    mean_epe64_list = []

    pck_1_list, pck_3_list, pck_5_list = [], [], []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))

    pyramid = VGGPyramid(train=False).to(dist_util.dev())
    SIZE = None

    for i, data in pbar:
        if not i * settings.env.val_batch_size < settings.env.num_samples:
            break
        else:
            radius = 4
            data = {k: v.to(dist_util.dev()) for k, v in data.items() if hasattr(v, 'to')}

            (
                data,
                H_ori,
                W_ori,
                source,
                target,
                batch_ori,
                source_256,
                target_256,
                source_vis,
                target_vis,
                mask,
            ) = prepare_data(settings, batch_preprocessing, SIZE, data)

            feature_size = 64
            
            with th.no_grad():
                raw_corr, trg_feat, src_feat = extract_raw_features(
                    pyramid, target, source, target_256, source_256, feature_size=feature_size
                )

            init_flow = correlation_to_flow_w_argmax(
                raw_corr.view(1, 1, feature_size, feature_size, feature_size, feature_size),
                output_shape=(feature_size, feature_size),
            )  # B, C, 55, 55

            vgg_flow_64 = init_flow / feature_size
            vgg_flow_64 = F.interpolate(vgg_flow_64, size=(H_ori, W_ori), mode='bilinear', align_corners=True)
            vgg_flow_64 = th.clamp(vgg_flow_64, min=-1, max=1)
            vgg_flow_64[:, 0, :, :] = vgg_flow_64[:, 0, :, :] * W_ori
            vgg_flow_64[:, 1, :, :] = vgg_flow_64[:, 1, :, :] * H_ori
            vgg_flow_64 = vgg_flow_64.permute(0, 2, 3, 1)[mask]
            batch_ori_ = batch_ori.permute(0, 2, 3, 1)[mask]
            epe64 = th.sum((vgg_flow_64 - batch_ori_.to(vgg_flow_64.device)) ** 2, dim=1).sqrt()
            logger.info(f'\n[init flow from vgg 64] epe: {epe64.mean()}')

            init_flow = F.interpolate(init_flow, size=feature_size, mode='bilinear', align_corners=True)
            init_flow /= feature_size

            logger.info(f"Starting sampling with VGG Features")
            for j in range(1):
                sample = run_sample_lr(
                    settings,
                    logger,
                    diffusion,
                    model,
                    radius,
                    source,
                    target,
                    feature_size,
                    raw_corr,
                    trg_feat,
                    init_flow,
                )

                sample_ = F.interpolate(sample, size=(H_ori, W_ori), mode='bilinear', align_corners=True)
                sample_[:, 0, :, :] = sample_[:, 0, :, :] * W_ori
                sample_[:, 1, :, :] = sample_[:, 1, :, :] * H_ori

                sample_ = sample_.permute(0, 2, 3, 1)[mask]
                batch_ori_ = batch_ori.permute(0, 2, 3, 1)[mask]
                epe = th.sum((sample_ - batch_ori_.to(sample_.device)) ** 2, dim=1).sqrt()
                logger.info(f'iter: {i}, epe: {epe.mean()}')

            logger.info('Running super resolution')
            sample_sr = None
            for j in range(1):
                batch_ori, sample_sr, init_flow_sr = run_sample_sr(
                    settings, logger, diffusion_sr, model_sr, pyramid, data, sample, sample_sr
                )

                sample_ = F.interpolate(sample_sr, size=(H_ori, W_ori), mode='bilinear', align_corners=True)
                sample_[:, 0, :, :] = sample_[:, 0, :, :] * W_ori
                sample_[:, 1, :, :] = sample_[:, 1, :, :] * H_ori

                sample_ = sample_.permute(0, 2, 3, 1)[mask]
                batch_ori_ = batch_ori.permute(0, 2, 3, 1)[mask]
                epe = th.sum((sample_ - batch_ori_.to(sample_.device)) ** 2, dim=1).sqrt()
                logger.info(f'sr iter: {i}, epe: {epe.mean()}')

            sample = F.interpolate(sample_sr, size=(H_ori, W_ori), mode='bilinear', align_corners=True)
            sample[:, 0, :, :] = sample[:, 0, :, :] * W_ori
            sample[:, 1, :, :] = sample[:, 1, :, :] * H_ori
            init_flow = F.interpolate(init_flow_sr, size=(H_ori, W_ori), mode='bilinear', align_corners=True)
            init_flow[:, 0, :, :] = init_flow[:, 0, :, :] * W_ori
            init_flow[:, 1, :, :] = init_flow[:, 1, :, :] * H_ori
            
            if settings.env.visualize:
                visualize(sample, category, rate, name_dataset, i, batch_ori, source_vis, target_vis, mask)

            sample = sample.permute(0, 2, 3, 1)[mask]
            init_flow = init_flow.permute(0, 2, 3, 1)[mask]
            batch_ori = batch_ori.permute(0, 2, 3, 1)[mask]
            epe = th.sum((sample - batch_ori.to(sample.device)) ** 2, dim=1).sqrt()
            epe_init = th.sum((init_flow - batch_ori.to(sample.device)) ** 2, dim=1).sqrt()

            epe_all_list.append(epe.view(-1).cpu().numpy())
            epe_init_all_list.append(epe_init.view(-1).cpu().numpy())
            mean_epe_list.append(epe.mean().item())
            mean_epe_init_list.append(epe_init.mean().item())
            mean_epe64_list.append(epe64.mean().item())

            logger.info(f'EPE :{epe.mean().item()}\t|\tEPE_init :{epe_init.mean().item()}')

            pck_1_list.append(epe.le(1.0).float().mean().item())
            pck_3_list.append(epe.le(3.0).float().mean().item())
            pck_5_list.append(epe.le(5.0).float().mean().item())
            logger.info(
                f'PCK1 :{epe.le(1.0).float().mean().item()}\t|\tPCK1_init :{epe_init.le(1.0).float().mean().item()}'
            )
            logger.info(
                f'PCK3 :{epe.le(3.0).float().mean().item()}\t|\tPCK3_init :{epe_init.le(3.0).float().mean().item()}'
            )
            logger.info(
                f'PCK5 :{epe.le(5.0).float().mean().item()}\t|\tPCK5_init :{epe_init.le(5.0).float().mean().item()}'
            )

        logger.log(f"created {(i+1) * settings.env.val_batch_size} samples")

    epe = np.mean((mean_epe_list))
    raw_epe = np.mean((mean_epe_init_list))
    epe_64 = np.mean((mean_epe64_list))
    epe_all = np.concatenate(epe_all_list)
    logger.info("Validation EPE: %f" % epe)
    logger.info("Validation Init Flow for SR EPE: %f" % raw_epe)
    logger.info("Validation Raw 64 EPE: %f" % epe_64)

    PCK_1_mean = np.mean((pck_1_list))
    PCK_3_mean = np.mean((pck_3_list))
    PCK_5_mean = np.mean((pck_5_list))
    logger.info("Validation PCK1: %f" % PCK_1_mean)
    logger.info("Validation PCK3: %f" % PCK_3_mean)
    logger.info("Validation PCK5: %f" % PCK_5_mean)
    pck1_dataset = np.mean(epe_all <= 1)
    pck3_dataset = np.mean(epe_all <= 3)
    pck5_dataset = np.mean(epe_all <= 5)

    output = {
        'AEPE': np.mean(mean_epe_list),
        'AEPE_init': np.mean(mean_epe_init_list),
        'AEPE_VGG64': np.mean(mean_epe64_list),
        'PCK_1_per_image': np.mean(pck_1_list),
        'PCK_3_per_image': np.mean(pck_3_list),
        'PCK_5_per_image': np.mean(pck_5_list),
        'PCK_1_per_dataset': pck1_dataset,
        'PCK_3_per_dataset': pck3_dataset,
        'PCK_5_per_dataset': pck5_dataset,
        'num_pixels_pck_1': np.sum(epe_all <= 1).astype(np.float64),
        'num_pixels_pck_3': np.sum(epe_all <= 3).astype(np.float64),
        'num_pixels_pck_5': np.sum(epe_all <= 5).astype(np.float64),
        'num_valid_corr': len(epe_all),
    }

    return_tuple = (epe, raw_epe, PCK_1_mean, PCK_3_mean, PCK_5_mean)
    logger.info(output)

    return return_tuple, output
