## DiffMatch: Diffusion Model for Dense Matching (ICLR'24 Oral)
This is the official implementation of the paper "Diffusion Model for Dense Matching" by Jisu Nam, Gyuseong Lee, Sunwoo Kim, Hyeonsu Kim, Hyoungwon Cho, Seyeon Kim and Seungryong Kim. \
\
For more information, check out the paper on [[arXiv](https://arxiv.org/abs/2305.19094)] and the [[project page](https://ku-cvlab.github.io/DiffMatch/)]. 

# Overall Architecture

Our model DiffMatch is illustrated below:


![Overall](/images/overall_architecture.png)


# Environment Settings

```
git clone https://github.com/KU-CVLAB/DiffMatch.git
cd DiffMatch

conda create -n diffmatch_env python=3.9
conda activate diffmatch_env

conda install gxx_linux-64
conda install -c conda-forge mpi4py tensorboardx
pip install -r requirements.txt

cd robustness/ImageNet-C/imagenet_c
pip install -e .
```

Create admin/local.py by running the following command and update the paths to the dataset. We provide an example admin/local_example_dped.py and local_example_coco.py for training on DPED and DPED+COCO, respectively, where all datasets are stored in data/.

```
python -c "from admin.environment import create_default_local_file; create_default_local_file()"
```

<!-- 
# Train

![alt text](/images/Train.png) -->

<!-- 
# Inference

![alt text](/images/Inference.png) -->


<!-- - Download pre-trained weights on [Link](https://drive.google.com/drive/folders/11kP1z0AmAl-Cb_MTLG7ViC3EHVoPZgHd?usp=sharing) -->

# Pre-trained weights

Download pre-trained weights on [Link](https://drive.google.com/drive/folders/1Ob-_LhH_AcaYanD-LyhEM4EJ3tslQHzr?usp=sharing).

# Training
Refer to admin/local_example_dped.py for training on DPED, and to admin/local_example_coco.py for training on DPED + COCO. To fine-tune the model for super-resolution, change the train_mode in admin/local.py from 'stage_1' to 'sr'.

      sh run_training.sh

# Inference
Refer to admin/local_example_dped.py for inference on HPatches, and to admin/local_example_coco.py for inference on ETH3D.

Inference on HPatches and ETH3D :

      sh run_sampling.sh

Inference on [ImageNet-C](https://github.com/hendrycks/robustness) corrupted HPatches and ETH3D :

      sh run_sampling_corrupt.sh

# Results

Qualitative results on HPatches :

![HPatches](/images/hp.png)

Qualitative results on ETH3D :

![ETH3D](/images/eth3d.png)

Qualitative results on HPatches using corruptions in ImageNet-C :

![HPatches Perturbed](/images/hp_perturb.png)

Qualitative results on ETH3D using corruptions in ImageNet-C : 

![ETH3D PErturbed](/images/eth3d_perturb.png)


# Acknowledgement <a name="Acknowledgement"></a>

We borrow code from public projects (huge thanks to all the projects). We mainly borrow code from [Improved DDPM](https://github.com/openai/improved-diffusion), [Dense Matching](https://github.com/PruneTruong/DenseMatching) and [ImageNet-C](https://github.com/hendrycks/robustness).

### BibTeX
If you find this research useful, please consider citing:
````BibTeX
@article{nam2023diffmatch,
  title={DiffMatch: Diffusion Model for Dense Matching},
  author={Nam, Jisu and Lee, Gyuseong and Kim, Sunwoo and Kim, Hyeonsu and Cho, Hyoungwon and Kim, Seyeon and Kim, Seungryong},
  journal={arXiv preprint arXiv:2305.19094},
  year={2023}
}
````


