import argparse
import importlib
import os
import random
from datetime import date
from shutil import copyfile

import cv2 as cv
import numpy as np
import torch
import torch.backends.cudnn

import admin.settings as ws_settings


def run_sampling(train_module, train_name, seed, name, cudnn_benchmark=True, corruption=False):
    """Run a sampling scripts in train_settings.
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """

    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    # dd/mm/YY
    today = date.today()
    d1 = today.strftime("%d/%m/%Y")
    print('Sampling:  {}  {}\nDate: {}'.format(train_module, train_name, d1))

    settings = ws_settings.Settings()
    settings.module_name = train_module
    settings.script_name = train_name
    settings.project_path = 'train_settings/{}/{}'.format(train_module, train_name)
    settings.seed = seed
    settings.name = name

    # will save the checkpoints there

    save_dir = os.path.join(settings.env.workspace_dir, settings.project_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    copyfile(settings.project_path + '.py', os.path.join(save_dir, settings.script_name + '.py'))

    expr_module = importlib.import_module('train_settings.{}.{}'.format(train_module.replace('/', '.'),
                                                                        train_name.replace('/', '.')))
    expr_func = getattr(expr_module, 'run')
    
    if corruption:
        for severity in [5]:
            settings.severity = severity
            for corruption_number in range(0, 15):
                # [0, 18]; useful for easy looping; 15, 16, 17, 18 are validation corruption numbers
                settings.corruption_number = corruption_number
                expr_func(settings)
    else:
        settings.severity = 0
        settings.corruption_number = 0
        expr_func(settings)



def main():
    parser = argparse.ArgumentParser(description='Run a sampling scripts in train_settings.')
    parser.add_argument('--train_module', type=str, help='Name of module in the "train_settings/" folder.')
    parser.add_argument('--train_name', type=str, help='Name of the train settings file.')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True,
                        help='Set cudnn benchmark on (1) or off (0) (default is on).')
    parser.add_argument('--seed', type=int, default=1992, help='Pseudo-RNG seed')
    parser.add_argument('--name', type=str, default="Default", help='Name of the experiment')
    parser.add_argument('--corruption', action='store_true')

    args = parser.parse_args()

    args.seed = random.randint(0, 3000000)
    args.seed = torch.initial_seed() & (2 ** 32 - 1)
    print('Seed is {}'.format(args.seed))
    random.seed(int(args.seed))
    np.random.seed(args.seed)
    

    run_sampling(
        args.train_module, args.train_name, cudnn_benchmark=args.cudnn_benchmark, seed=args.seed, 
        name=args.name, corruption=args.corruption)


if __name__ == '__main__':
    main()