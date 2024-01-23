import torch
import torchvision.transforms as transforms
from termcolor import colored

from datasets.load_pre_made_datasets.load_pre_made_dataset import \
    PreMadeDataset
from datasets.object_augmented_dataset import (
    MSCOCO, AugmentedImagePairsDatasetMultipleObjects)
from datasets.object_augmented_dataset.synthetic_object_augmentation_for_pairs_multiple_ob import \
    RandomAffine
from training.actors.batch_processing import (CATsBatchPreprocessing,
                                              GLUNetBatchPreprocessing)
from utils_data.image_transforms import ArrayToTensor
from utils_data.loaders import Loader

from .improved_diffusion import dist_util, logger
from .improved_diffusion.resample import create_named_schedule_sampler
from .improved_diffusion.script_util import (args_to_dict,
                                             create_model_and_diffusion,
                                             model_and_diffusion_defaults)
from .improved_diffusion.train_util import TrainLoop


def run(settings):
    settings.description = 'train settings for DiffMatch'

    dist_util.setup_dist()
    torch.cuda.set_device(dist_util.dev())
    logger.configure(dir=f"{settings.env.train_mode}_{settings.env.dataset_name}")
                     
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        device=dist_util.dev(),
        train_mode=settings.env.train_mode,
        **args_to_dict(settings, model_and_diffusion_defaults().keys())
    )
    
    if settings.env.resume_checkpoint:
        model.load_state_dict(
                        dist_util.load_state_dict(
                            settings.env.resume_checkpoint, map_location='cpu'
                        ), strict=False
                    )

    settings.device = dist_util.dev()
    print(f"Setting device to {settings.device}")
    model = model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(settings.env.schedule_sampler, diffusion)
    
    logger.log("creating data loader...")    
    
    # 1. Define training and validation datasets
    # datasets, pre-processing of the images is done within the network function !
    if settings.env.dataset_name == 'COCO2014':
        # dataset parameters
        # independently moving objects
        settings.nbr_objects = 1
        settings.min_area_objects = 1300
        settings.compute_object_reprojection_mask = True
        # very important, we compute the object reprojection mask, will be used for training

        # perturbations
        perturbations_parameters_v2 = {'elastic_param': {"max_sigma": 0.04, "min_sigma": 0.1, "min_alpha": 1,
                                                        "max_alpha": 0.4},
                                    'max_sigma_mask': 10, 'min_sigma_mask': 3}

        # Train dataset: synthetic data with perturbations + independently moving objects
        # object foreground dataset
        fg_tform = RandomAffine(p_flip=0.0, max_rotation=30.0,
                                max_shear=0, max_ar_factor=0.,
                                max_scale=0.3, pad_amount=0)

        coco_dataset_train = MSCOCO(root=settings.env.coco, split='train', version='2014',
                                    min_area=settings.min_area_objects)

        # base dataset with image pairs and ground-truth flow field + adding perturbations
        train_dataset, _ = PreMadeDataset(root=settings.env.training_cad_520,
                                        source_image_transform=None,
                                        target_image_transform=None,
                                        flow_transform=None,
                                        co_transform=None,
                                        split=1,
                                        get_mapping=False,
                                        add_discontinuity=True,
                                        parameters_v2=perturbations_parameters_v2,
                                        max_nbr_perturbations=15,
                                        min_nbr_perturbations=5)  # only training

        # add independently moving objects + compute the reprojection mask
        # datasets, pre-processing of the images is done within the network function !
        source_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
        target_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
        flow_transform = transforms.Compose([ArrayToTensor()])  # just put channels first and put it to float
        co_transform = None
        # co_transform = Scale((512, 512))
        # base training data is DPED-CityScape-ADE + 1 object from COCO
        train_dataset, _ = PreMadeDataset(root=settings.env.training_cad_520,
                                        source_image_transform=None,
                                        target_image_transform=None,
                                        flow_transform=None,
                                        co_transform=None,
                                        split=1)  # only training

        # we then adds the object on the dataset
        train_dataset = AugmentedImagePairsDatasetMultipleObjects(foreground_image_dataset=coco_dataset_train,
                                                                background_image_dataset=train_dataset,
                                                                foreground_transform=fg_tform,
                                                                number_of_objects=1, object_proba=0.9,
                                                                source_image_transform=source_img_transforms,
                                                                target_image_transform=target_img_transforms,
                                                                flow_transform=flow_transform,
                                                                co_transform=co_transform)
        # dataloader
        train_loader = Loader('train', train_dataset, batch_size=settings.env.batch_size, shuffle=True,
                            drop_last=False, training=True, num_workers=settings.env.n_threads)
    
    elif settings.env.dataset_name == 'DPED':
        img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
        flow_transform = transforms.Compose([ArrayToTensor()])  # just put channels first and put it to float
        co_transform = None

        # base training data is DPED-CityScape-ADE
        train_dataset, _ = PreMadeDataset(root=settings.env.training_cad_520,
                                        source_image_transform=img_transforms,
                                        target_image_transform=img_transforms,
                                        flow_transform=flow_transform,
                                        co_transform=co_transform,
                                        split=1,
                                        get_mapping=False)

        # validation dataset
        _, val_dataset = PreMadeDataset(root=settings.env.validation_cad_520,
                                        source_image_transform=img_transforms,
                                        target_image_transform=img_transforms,
                                        flow_transform=flow_transform,
                                        co_transform=co_transform,
                                        split=0)

        # 2. Define dataloaders
        train_loader = Loader('train', train_dataset, batch_size=settings.env.batch_size, shuffle=True,
                            drop_last=False, training=True, num_workers=settings.env.n_threads)
   
    # Setting dataset name into diffusion because of the semantic setting.
    setattr(diffusion, 'dataset', settings.env.dataset_name)

    # but better results are obtained with using simple bilinear interpolation instead of deconvolutions.
    print(colored('==> ', 'blue') + 'model created.')

    logger.log("training...")
    
    # 4. Define batch_processing
    if settings.env.dataset_name == 'COCO2014':
        batch_preprocessing = CATsBatchPreprocessing(settings, apply_mask=False, apply_mask_zero_borders=False,
                                                sparse_ground_truth=False)
    else:
        batch_preprocessing = GLUNetBatchPreprocessing(settings, apply_mask=False, apply_mask_zero_borders=False,
                                                sparse_ground_truth=False, megadepth=False)
    # 5, Define loss module    
    TrainLoop(
        model=model,
        diffusion=diffusion,
        settings=settings,
        batch_preprocessing=batch_preprocessing,
        data=train_loader,
        batch_size=settings.env.batch_size,
        microbatch=settings.env.microbatch,
        lr=settings.env.lr,
        ema_rate=settings.env.ema_rate,
        log_interval=settings.env.log_interval,
        save_interval=settings.env.save_interval,
        resume_checkpoint=settings.env.resume_checkpoint,
        use_fp16=settings.env.use_fp16,
        fp16_scale_growth=settings.env.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=settings.env.weight_decay,
        lr_anneal_steps=settings.env.lr_anneal_steps,

    ).run_loop()





