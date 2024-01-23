import importlib
import os
from collections import OrderedDict


def create_default_local_file():
    """ Contains the path to all necessary datasets or useful folders (like workspace, pretrained models..)"""
    path = os.path.join(os.path.dirname(__file__), 'local.py')

    empty_str = '\'\'' 
    default_settings = OrderedDict({
        'workspace_dir': empty_str,
        'tensorboard_dir': 'self.workspace_dir',
        'pretrained_networks': 'self.workspace_dir',
        'pre_trained_models_dir' : empty_str,
        'hp': empty_str,
        'eth3d': empty_str,
        'training_cad_520': empty_str,
        'validation_cad_520': empty_str,
        'coco': empty_str,
        'dataset_name': empty_str, 
        'nbr_objects': 4,
        'min_area_objects': 1300,
        'compute_object_reprojection_mask': True,
        'n_threads': 16, 
        'initial_pretrained_model': None,
        'data_dir': empty_str, 
        'schedule_sampler': "'uniform'",
        'lr': empty_str, 
        'weight_decay': 0.0,
        'lr_anneal_steps': 0,
        'batch_size': 18, 
        'microbatch': -1,
        'ema_rate': 0.9999,
        'log_interval': 10,
        'save_interval': 5000,
        'resume_checkpoint': empty_str,
        'train_mode': empty_str,
        'use_fp16': False,
        'fp16_scale_growth': 1e-3,
        'image_size': 64,
        'flow_size': (64,64),
        'num_channels': 128,
        'num_res_blocks': 3,
        'num_heads': 4,
        'num_heads_upsample': -1,
        'attention_resolutions': '"16,8"',
        'dropout': 0.0,
        'learn_sigma': False,
        'sigma_small': False,
        'class_cond': False,
        'diffusion_steps': 5, 
        'noise_schedule': "'cosine'",
        'use_kl': False,
        'predict_xstart': True,
        'rescale_timesteps': True,
        'rescale_learned_sigmas': True,
        'use_checkpoint': False,
        'use_scale_shift_norm': True, 
        'clip_denoised': False,
        'num_samples': 10000,
        'val_batch_size': 1,
        'use_ddim': False,
        'model_path': empty_str,
        'model_path_sr': empty_str,
        'timestep_respacing': "''",
        'eval_dataset': empty_str, 
        'n_batch': empty_str,
        'visualize': empty_str
        })

    comment = {'workspace_dir': 'Base directory for saving network checkpoints.',
               'tensorboard_dir': 'Directory for tensorboard files.',
               'dataset_name': 'Training dataset name ("DPED" or "COCO2014")',
               'lr': 'learning rate for training (recommendation: 3e-5 for DPED and 1e-4 for COCO)',
               'train_mode': 'Training mode ("stage_1" or "sr")',
               'model_path': 'Pre-trained model path for evaluation',
               'model_path_sr': 'Pre-trained super-resolution model path for evaluation',
               'eval_dataset': 'Evaluation dataset ("hp" or "eth3d")',
               'n_batch': 'The number of multiple hypotheses',
               'visualize': 'Set True, if you want qualitative results.'}
    

    with open(path, 'w') as f:
        f.write('class EnvironmentSettings:\n')
        f.write('    def __init__(self):\n')

        for attr, attr_val in default_settings.items():
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            if comment_str is None:
                f.write('        self.{} = {}\n'.format(attr, attr_val))
            else:
                f.write('        self.{} = {}    # {}\n'.format(attr, attr_val, comment_str))


def env_settings():
    env_module_name = 'admin.local'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.EnvironmentSettings()
    except:
        env_file = os.path.join(os.path.dirname(__file__), 'local.py')

        create_default_local_file()
        raise RuntimeError('YOU HAVE NOT SETUP YOUR local.py!!!\n Go to "{}" and set all the paths you need. '
                           'Then try to run again.'.format(env_file))
