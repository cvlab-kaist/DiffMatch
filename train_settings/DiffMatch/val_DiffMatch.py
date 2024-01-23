import os

import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import datasets
from utils_data.image_transforms import ArrayToTensor

from .evaluation import run_evaluation_eth3d, run_evaluation_generic
from .improved_diffusion import dist_util, logger
from .improved_diffusion.script_util import (args_to_dict,
                                             create_model_and_diffusion,
                                             model_and_diffusion_defaults)

corruption = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
    'speckle_noise',
    'gaussian_blur',
    'spatter',
    'saturate',
]

def run(settings):
    dist_util.setup_dist()
    severity = settings.severity
    corruption_number = settings.corruption_number
    if severity > 0:
        logger.configure(
            dir=f"SAMPLING_{settings.env.eval_dataset}_{settings.name}_{corruption[corruption_number]}_sev{severity}"
        )
        logger.log(f"severity: {severity}, corruption_number: {corruption_number}")
        logger.log(f"Evaluating on corrupted dataset with {corruption[corruption_number]} severity {severity}.")
    else:
        logger.configure(dir=f"SAMPLING_{settings.env.eval_dataset}_{settings.name}")
        logger.log(f"Corruption Disabled. Evaluating on Original {settings.env.eval_dataset}")
    
    logger.log("Loading model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        device=dist_util.dev(),
        train_mode=settings.env.train_mode,
        **args_to_dict(settings, model_and_diffusion_defaults().keys()),
    )
    setattr(diffusion, "settings", settings)

    model.cpu().load_state_dict(dist_util.load_state_dict(settings.env.model_path, map_location="cpu"), strict=False)
    logger.log(f"Model loaded with {settings.env.model_path}")

    model_sr, diffusion_sr = create_model_and_diffusion(
        device=dist_util.dev(), train_mode='sr', **args_to_dict(settings, model_and_diffusion_defaults().keys())
    )
    setattr(diffusion_sr, "settings", settings)

    model_sr.cpu().load_state_dict(
        dist_util.load_state_dict(settings.env.model_path_sr, map_location="cpu"), strict=False
    )
    logger.log(f"Model_sr loaded with {settings.env.model_path_sr}")

    model.to(dist_util.dev())
    model.eval()

    model_sr.to(dist_util.dev())
    model_sr.eval()

    logger.log("Creating data loader...")
    logger.info('\n:============== Logging Configs ==============')
    for key, value in settings.env.__dict__.items():
        if key in ['model_path', 'timestep_respacing', 'eval_dataset']:
            logger.info(f"\t{key}:\t{value}")
    logger.info(':===============================================\n')

    # 1. Define training and validation datasets
    co_transform = None
    target_transform = transforms.Compose([ArrayToTensor()])  # only put channel first
    input_transform = transforms.Compose([ArrayToTensor(get_float=False)])  # only put channel first

    if 'hp' in settings.env.eval_dataset:
        original_size = True
        if settings.env.eval_dataset == 'hp-240':
            original_size = False
        number_of_scenes = 5 + 1
        list_of_outputs = []
        output_dict_list = []

        # loop over scenes (1-2, 1-3, 1-4, 1-5, 1-6)
        for id, k in enumerate(range(2, number_of_scenes + 2)):
            if id != 5:
                _, test_set = datasets.HPatchesdataset(
                    settings.env.hp,
                    os.path.join('assets', 'hpatches_1_{}.csv'.format(k)),
                    input_transform,
                    target_transform,
                    co_transform,
                    use_original_size=original_size,
                    split=0,
                    severity=severity, 
                    corruption_number=corruption_number
                )
                logger.info(f"Starting sampling {os.path.join('assets', 'hpatches_1_{}.csv'.format(k)),}")
            else:
                break
            val_loader = DataLoader(test_set, batch_size=1, num_workers=8)
            (
                output_scene,
                raw_output_scene,
                PCK_1_scene,
                PCK_3_scene,
                PCK_5_scene,
            ), output_dict = run_evaluation_generic(
                settings, logger, val_loader, diffusion, model, diffusion_sr, model_sr, id, id
            )

            list_of_outputs.append(output_scene)
            logger.info(output_scene)
            logger.info(raw_output_scene)
            logger.info(PCK_1_scene)
            logger.info(PCK_3_scene)
            logger.info(PCK_5_scene)
            output_dict_list.append(output_dict)

        output = {
            'scene_1': list_of_outputs[0],
            'scene_2': list_of_outputs[1],
            'scene_3': list_of_outputs[2],
            'scene_4': list_of_outputs[3],
            'scene_5': list_of_outputs[4],
        }
        for scene_idx, _out_dict in enumerate(output_dict_list):
            logger.info(f"--------------------------------------------")
            logger.info(f"Scene {scene_idx + 1}")
            for k, v in _out_dict.items():
                logger.info(f"{k}: {v}")

        logger.info(output)

    elif 'eth3d' in settings.env.eval_dataset:
        output = run_evaluation_eth3d(
            settings,
            settings.env.eth3d,
            input_transform,
            target_transform,
            co_transform,
            logger,
            diffusion,
            model,
            diffusion_sr,
            model_sr,
            severity=severity,
            corruption_number=corruption_number,
        )
        logger.info(f"{k}: {v}\n" for k, v in output.items())
        logger.info(output)

    else:
        raise ValueError(f"Unknown dataset {settings.env.eval_dataset}")

    dist.barrier()
    logger.log("sampling complete")
