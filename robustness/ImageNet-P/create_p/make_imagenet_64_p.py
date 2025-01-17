import ctypes
import os
from io import BytesIO
from shutil import rmtree
from tempfile import gettempdir

import cv2
import numpy as np
import skimage.color as skcolor
import torch
import torchvision.transforms as trn
import torchvision.transforms.functional as trn_F
from PIL import Image as PILImage
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian
from skimage.util import random_noise
from wand.api import library as wandlibrary
from wand.image import Image as WandImage

imagenet_val_folder = '/share/data/vision-greg/ImageNet/clsloc/images/val/'

# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def brightness(_x, c=0.):
    _x = np.array(_x, copy=True) / 255.
    _x = skcolor.rgb2hsv(_x)
    _x[:, :, 2] = np.clip(_x[:, :, 2] + c, 0, 1)
    _x = skcolor.hsv2rgb(_x)

    return np.uint8(_x * 255)


for index, folder in enumerate(sorted(os.listdir(imagenet_val_folder))):
    if index < 750 or index > 1000: continue
    
    for img_loc in os.listdir(imagenet_val_folder + folder):
        orig_img = PILImage.open(imagenet_val_folder + folder + '/' + img_loc).convert('RGB')
        img = trn.Resize((64, 64))(orig_img)

        # /////////////// Test Data ///////////////

        # # /////////////// Gaussian Noise Code ///////////////
        #
        # tmp = os.path.join(gettempdir(), '.{}'.format(hash(os.times())))
        # os.makedirs(tmp)
        #
        # x = trn.Compose([trn.CenterCrop(64), trn.ToTensor()])(img)
        #
        # z = trn.ToPILImage()(x)
        # z.save(os.path.join(tmp, 'img0.png'))
        #
        # # for i in range(1, 31):
        # #     z = trn.ToPILImage()(torch.clamp(x + 0.025 * torch.randn_like(x), 0, 1))
        # #     z.save(os.path.join(tmp, 'img' + str(i) + '.png'))
        # # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/gaussian_noise/' + folder + '/'
        #
        # # for i in range(1, 31):
        # #     z = trn.ToPILImage()(torch.clamp(x + 0.05 * torch.randn_like(x), 0, 1))
        # #     z.save(os.path.join(tmp, 'img' + str(i) + '.png'))
        # # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/gaussian_noise_2/' + folder + '/'
        #
        # # for i in range(1, 31):
        # #     z = trn.ToPILImage()(torch.clamp(x + 0.075 * torch.randn_like(x), 0, 1))
        # #     z.save(os.path.join(tmp, 'img' + str(i) + '.png'))
        # # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/gaussian_noise_3/' + folder + '/'
        #
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        #
        # os.system("./ffmpeg/ffmpeg -r 1 -framerate 1 -i " + os.path.join(tmp, 'img') + \
        #           "%01d.png -vcodec libx264 -tune grain -preset veryslow -y " + \
        #           save_dir + img_loc[:-5] + ".mp4")
        #
        # rmtree(tmp, ignore_errors=True)
        #
        # # /////////////// End Gaussian Noise Code ///////////////

        # # /////////////// Shot Noise Code ///////////////
        #
        # tmp = os.path.join(gettempdir(), '.{}'.format(hash(os.times())))
        # os.makedirs(tmp)
        #
        # x = trn.CenterCrop(64)(img)
        #
        # x.save(os.path.join(tmp, 'img0.png'))
        #
        # # for i in range(1, 31):
        # #     z = np.array(x, copy=True) / 255.
        # #     z = PILImage.fromarray(
        # #         np.uint8(255 * np.clip(np.random.poisson(z * 500) / 500., 0, 1)))
        # #     z.save(os.path.join(tmp, 'img' + str(i) + '.png'))
        # # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/shot_noise/' + folder + '/'
        #
        # # for i in range(1, 31):
        # #     z = np.array(x, copy=True) / 255.
        # #     z = PILImage.fromarray(
        # #         np.uint8(255 * np.clip(np.random.poisson(z * 250) / 250., 0, 1)))
        # #     z.save(os.path.join(tmp, 'img' + str(i) + '.png'))
        # # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/shot_noise_2/' + folder + '/'
        #
        # # for i in range(1, 31):
        # #     z = np.array(x, copy=True) / 255.
        # #     z = PILImage.fromarray(
        # #         np.uint8(255 * np.clip(np.random.poisson(z * 125) / 125., 0, 1)))
        # #     z.save(os.path.join(tmp, 'img' + str(i) + '.png'))
        # # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/shot_noise_3/' + folder + '/'
        #
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        #
        # os.system("./ffmpeg/ffmpeg -r 1 -framerate 1 -i " + os.path.join(tmp, 'img') + \
        #           "%01d.png -vcodec libx264 -tune grain -preset veryslow -y " + \
        #           save_dir + img_loc[:-5] + ".mp4")
        #
        # rmtree(tmp, ignore_errors=True)
        #
        # # /////////////// End Shot Noise Code ///////////////

        # /////////////// Motion Blur Code ///////////////

        tmp = os.path.join(gettempdir(), '.{}'.format(hash(os.times())))
        os.makedirs(tmp)

        for i in range(0,31):

            z = orig_img
            output = BytesIO()
            z.save(output, format='PNG')
            z = MotionImage(blob=output.getvalue())

            z.motion_blur(radius=14, sigma=3, angle=(i-30)*5)

            z = cv2.imdecode(np.fromstring(z.make_blob(), np.uint8),
                             cv2.IMREAD_UNCHANGED)

            if z.shape != (64, 64):
                z = np.clip(z[..., [2, 1, 0]], 0, 255)  # BGR to RGB
            else:  # grayscale to RGB
                z = np.clip(np.array([z, z, z]).transpose((1, 2, 0)), 0, 255)

            trn.Resize(64)(PILImage.fromarray(z)).save(os.path.join(tmp, 'img' + str(i) + '.png'))

        save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/motion_blur/' + folder + '/'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        os.system("./ffmpeg/ffmpeg -r 1 -framerate 1 -i " + os.path.join(tmp, 'img') + \
                  "%01d.png -vcodec libx264 -crf 20 -preset veryslow -y " + \
                  save_dir + img_loc[:-5] + ".mp4")

        rmtree(tmp, ignore_errors=True)

        # /////////////// End Motion Blur Code ///////////////

        # # /////////////// Zoom Blur Code ///////////////
        #
        # tmp = os.path.join(gettempdir(), '.{}'.format(hash(os.times())))
        # os.makedirs(tmp)
        #
        # z = trn.CenterCrop(64)(img)
        # avg = trn.ToTensor()(z)
        # z.save(os.path.join(tmp, 'img0.png'))
        # for i in range(1, 31):
        #     z = trn.CenterCrop(64)(trn_F.affine(img, angle=0, translate=(0, 0),
        #                                         scale=1+0.004*i, shear=0, resample=PILImage.BILINEAR))
        #     avg += trn.ToTensor()(z)
        #     trn.ToPILImage()(avg / (i + 1)).save(os.path.join(tmp, 'img' + str(i) + '.png'))
        #
        # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/zoom_blur/' + folder + '/'
        #
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        #
        # os.system("./ffmpeg/ffmpeg -r 1 -framerate 1 -i " +
        #           os.path.join(tmp, 'img') + "%01d.png -vcodec libx264 -crf 20 -preset veryslow -y " +
        #           save_dir + img_loc[:-5] + ".mp4")
        #
        # rmtree(tmp, ignore_errors=True)
        #
        # # /////////////// End Zoom Blur Code ///////////////

        # # /////////////// Snow Code ///////////////
        #
        # tmp = os.path.join(gettempdir(), '.{}'.format(hash(os.times())))
        # os.makedirs(tmp)
        #
        # x = trn.CenterCrop(64)(img)
        # x = np.array(x) / 255.
        #
        # snow_layer = np.random.normal(size=(64, 64), loc=0.05, scale=0.28)
        #
        # snow_layer = clipped_zoom(snow_layer[..., np.newaxis], 2)
        # snow_layer[snow_layer < 0.5] = 0
        #
        # snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
        # output = BytesIO()
        # snow_layer.save(output, format='PNG')
        # output = output.getvalue()
        #
        # for i in range(0, 31):
        #     moving_snow = MotionImage(blob=output)
        #     moving_snow.motion_blur(radius=10, sigma=2.5, angle=i*4-150)
        #
        #     snow_layer = cv2.imdecode(np.fromstring(moving_snow.make_blob(), np.uint8),
        #                               cv2.IMREAD_UNCHANGED) / 255.
        #     snow_layer = snow_layer[..., np.newaxis]
        #
        #     z = 0.85 * x + (1 - 0.85) * np.maximum(
        #         x, cv2.cvtColor(np.float32(x), cv2.COLOR_RGB2GRAY).reshape(64, 64, 1) * 1.5 + 0.5)
        #
        #     z = np.uint8(np.clip(z + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255)
        #
        #     PILImage.fromarray(z).save(os.path.join(tmp, 'img' + str(i) + '.png'))
        #
        # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/snow/' + folder + '/'
        #
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        #
        # os.system("./ffmpeg/ffmpeg -r 1 -framerate 1 -i " + os.path.join(tmp, 'img') + \
        #           "%01d.png -vcodec libx264 -crf 20 -preset veryslow -y " + \
        #           save_dir + img_loc[:-5] + ".mp4")
        #
        # rmtree(tmp, ignore_errors=True)
        #
        # # /////////////// End Snow Code ///////////////

        # # /////////////// Brightness Code ///////////////
        #
        # tmp = os.path.join(gettempdir(), '.{}'.format(hash(os.times())))
        # os.makedirs(tmp)
        #
        # x = trn.CenterCrop(64)(img)
        #
        # for i in range(0, 31):
        #     z = PILImage.fromarray(brightness(x, c=(i - 15) * 2 / 100.))
        #     z.save(os.path.join(tmp, 'img' + str(i) + '.png'))
        #
        # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/brightness/' + folder + '/'
        #
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        #
        # os.system("./ffmpeg/ffmpeg -r 1 -framerate 1 -i " + os.path.join(tmp, 'img') + \
        #           "%01d.png -vcodec libx264 -crf 20 -preset veryslow -y " + \
        #           save_dir + img_loc[:-5] + ".mp4")
        #
        # rmtree(tmp, ignore_errors=True)
        #
        # # /////////////// End Brightness Code ///////////////

        # # /////////////// Translate Code ///////////////
        #
        # tmp = os.path.join(gettempdir(), '.{}'.format(hash(os.times())))
        # os.makedirs(tmp)
        #
        # for i in range(0,31):
        #     z = trn.CenterCrop(64)(trn_F.affine(img, angle=0, translate=(i-15, 0), scale=1, shear=0))
        #     z.save(os.path.join(tmp, 'img'+str(i)+'.png'))
        #
        # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/translate/' + folder + '/'
        #
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        #
        # os.system("./ffmpeg/ffmpeg -r 1 -framerate 1 -i " + os.path.join(tmp, 'img') +
        #           "%01d.png -vcodec libx264 -crf 20 -preset veryslow -y " +
        #           save_dir + img_loc[:-5] + ".mp4")
        #
        # rmtree(tmp, ignore_errors=True)
        #
        # # /////////////// End Translate Code ///////////////

        # # /////////////// Rotate Code ///////////////
        #
        # tmp = os.path.join(gettempdir(), '.{}'.format(hash(os.times())))
        # os.makedirs(tmp)
        #
        # for i in range(0, 31):
        #     z = trn.CenterCrop(64)(trn_F.affine(img, angle=i-15, translate=(0, 0),
        #                                         scale=1., shear=0, resample=PILImage.BILINEAR))
        #     z.save(os.path.join(tmp, 'img' + str(i) + '.png'))
        #
        # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/rotate/' + folder + '/'
        #
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        #
        # os.system("./ffmpeg/ffmpeg -r 1 -framerate 1 -i " +
        #           os.path.join(tmp, 'img') + "%01d.png -vcodec libx264 -crf 20 -preset veryslow -y " +
        #           save_dir + img_loc[:-5] + ".mp4")
        #
        # rmtree(tmp, ignore_errors=True)
        #
        # # /////////////// End Rotate Code ///////////////

        # # /////////////// Tilt Code ///////////////
        #
        # tmp = os.path.join(gettempdir(), '.{}'.format(hash(os.times())))
        # os.makedirs(tmp)
        #
        # x = np.array(img)
        # h, w = x.shape[0:2]
        #
        # for i in range(0, 31):
        #     phi, theta = np.deg2rad(i-15), np.deg2rad(i-15)
        #
        #     f = np.sqrt(w ** 2 + h ** 2)
        #
        #     P1 = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1], [0, 0, 1]])
        #
        #     RX = np.array([[1, 0, 0, 0], [0, np.cos(theta), -np.sin(theta), 0],
        #                    [0, np.sin(theta), np.cos(theta), 0], [0, 0, 0, 1]])
        #
        #     RY = np.array([[np.cos(phi), 0, -np.sin(phi), 0], [0, 1, 0, 0],
        #                    [np.sin(phi), 0, np.cos(phi), 0], [0, 0, 0, 1]])
        #
        #     T = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
        #                   [0, 0, 1, f], [0, 0, 0, 1]])
        #
        #     P2 = np.array([[f, 0, w / 2, 0], [0, f, h / 2, 0], [0, 0, 1, 0]])
        #
        #     mat = P2 @ T @ RX @ RY @ P1
        #
        #     z = trn.CenterCrop(64)(PILImage.fromarray(cv2.warpPerspective(x, mat, (w, h))))
        #
        #     z.save(os.path.join(tmp, 'img' + str(i) + '.png'))
        #
        # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/tilt/' + folder + '/'
        #
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        #
        # os.system("./ffmpeg/ffmpeg -r 1 -framerate 1 -i " + os.path.join(tmp, 'img') + \
        #           "%01d.png -vcodec libx264 -crf 20 -preset veryslow -y " + \
        #           save_dir + img_loc[:-5] + ".mp4")
        #
        # rmtree(tmp, ignore_errors=True)
        #
        # # /////////////// End Tilt Code ///////////////

        # # /////////////// Scale Code ///////////////
        #
        # tmp = os.path.join(gettempdir(), '.{}'.format(hash(os.times())))
        # os.makedirs(tmp)
        #
        # for i in range(0, 31):
        #     z = trn.CenterCrop(64)(trn_F.affine(img, angle=0, translate=(0, 0),
        #                                         scale=(i * 2.5 + 40) / 100., shear=0, resample=PILImage.BILINEAR))
        #     z.save(os.path.join(tmp, 'img' + str(i) + '.png'))
        #
        # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/scale/' + folder + '/'
        #
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        #
        # os.system("./ffmpeg/ffmpeg -r 1 -framerate 1 -i " +
        #           os.path.join(tmp, 'img') + "%01d.png -vcodec libx264 -crf 20 -preset veryslow -y " +
        #           save_dir + img_loc[:-5] + ".mp4")
        #
        # rmtree(tmp, ignore_errors=True)
        #
        # # /////////////// End Scale Code ///////////////

        # /////////////// Validation Data ///////////////

        # # /////////////// Speckle Noise Code ///////////////
        #
        # tmp = os.path.join(gettempdir(), '.{}'.format(hash(os.times())))
        # os.makedirs(tmp)
        #
        # x = trn.CenterCrop(64)(img)
        #
        # x.save(os.path.join(tmp, 'img0.png'))
        # x = np.array(x) / 255.
        #
        # # for i in range(1, 31):
        # #     z = PILImage.fromarray(
        # #         np.uint8(255 * np.clip(x + x * np.random.normal(size=x.shape, scale=0.05), 0, 1)))
        # #     z.save(os.path.join(tmp, 'img' + str(i) + '.png'))
        # # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/speckle_noise/' + folder + '/'
        #
        # # for i in range(1, 31):
        # #     z = PILImage.fromarray(
        # #         np.uint8(255 * np.clip(x + x * np.random.normal(size=x.shape, scale=0.1), 0, 1)))
        # #     z.save(os.path.join(tmp, 'img' + str(i) + '.png'))
        # # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/speckle_noise_2/' + folder + '/'
        #
        # # for i in range(1, 31):
        # #     z = PILImage.fromarray(
        # #         np.uint8(255 * np.clip(x + x * np.random.normal(size=x.shape, scale=0.15), 0, 1)))
        # #     z.save(os.path.join(tmp, 'img' + str(i) + '.png'))
        # # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/speckle_noise_3/' + folder + '/'
        #
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        #
        # os.system("./ffmpeg/ffmpeg -r 1 -framerate 1 -i " + os.path.join(tmp, 'img') + \
        #           "%01d.png -vcodec libx264 -tune grain -preset veryslow -y " + \
        #           save_dir + img_loc[:-5] + ".mp4")
        #
        # rmtree(tmp, ignore_errors=True)
        #
        # # /////////////// End Speckle Noise Code ///////////////

        # # /////////////// Gaussian Blur Code ///////////////
        #
        # tmp = os.path.join(gettempdir(), '.{}'.format(hash(os.times())))
        # os.makedirs(tmp)
        #
        # x = trn.CenterCrop(64)(img)
        # for i in range(0, 31):
        #     z = PILImage.fromarray(
        #         np.uint8(255*gaussian(np.array(x, copy=True)/255.,
        #                               sigma=0.35 + 0.014*i, multichannel=True, truncate=7.0)))
        #
        #     z.save(os.path.join(tmp, 'img' + str(i) + '.png'))
        #
        # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/gaussian_blur/' + folder + '/'
        #
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        #
        # os.system("./ffmpeg/ffmpeg -r 1 -framerate 1 -i " +
        #           os.path.join(tmp, 'img') + "%01d.png -vcodec libx264 -crf 20 -preset veryslow -y " +
        #           save_dir + img_loc[:-5] + ".mp4")
        #
        # rmtree(tmp, ignore_errors=True)
        #
        # # /////////////// End Gaussian Blur Code ///////////////

        # # /////////////// Spatter Code ///////////////
        #
        # tmp = os.path.join(gettempdir(), '.{}'.format(hash(os.times())))
        # os.makedirs(tmp)
        #
        # x = trn.CenterCrop(64)(img)
        # x = cv2.cvtColor(np.array(x, dtype=np.float32) / 255., cv2.COLOR_BGR2BGRA)
        #
        # liquid_layer = np.random.normal(size=x.shape[:2], loc=0.6, scale=0.25)
        # liquid_layer = gaussian(liquid_layer, sigma=1.75)
        # liquid_layer[liquid_layer < 0.7] = 0
        #
        # for i in range(0, 31):
        #
        #     liquid_layer_i = (liquid_layer * 255).astype(np.uint8)
        #     dist = 255 - cv2.Canny(liquid_layer_i, 50, 150)
        #     dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        #     _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        #     dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        #     dist = cv2.equalizeHist(dist)
        #     ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        #     dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        #     dist = cv2.blur(dist, (3, 3)).astype(np.float32)
        #
        #     m = cv2.cvtColor(liquid_layer_i * dist, cv2.COLOR_GRAY2BGRA)
        #     m /= np.max(m, axis=(0, 1))
        #     m *= 0.6
        #
        #     # water is pale turqouise
        #     color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
        #                             238 / 255. * np.ones_like(m[..., :1]),
        #                             238 / 255. * np.ones_like(m[..., :1])), axis=2)
        #
        #     color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        #
        #     z = np.uint8(cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255)
        #
        #     liquid_layer = np.apply_along_axis(lambda mat:
        #                                        np.convolve(mat, np.array([0.2, 0.8]), mode='same'),
        #                                        axis=0, arr=liquid_layer)
        #
        #     PILImage.fromarray(z).save(os.path.join(tmp, 'img' + str(i) + '.png'))
        #
        # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/spatter/' + folder + '/'
        #
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        #
        # os.system("./ffmpeg/ffmpeg -r 1 -framerate 1 -i " + os.path.join(tmp, 'img') + \
        #           "%01d.png -vcodec libx264 -crf 20 -preset veryslow -y " + \
        #           save_dir + img_loc[:-5] + ".mp4")
        #
        # rmtree(tmp, ignore_errors=True)
        #
        # # /////////////// End Spatter Code ///////////////

        # # /////////////// Shear Code ///////////////
        #
        # tmp = os.path.join(gettempdir(), '.{}'.format(hash(os.times())))
        # os.makedirs(tmp)
        #
        # for i in range(0, 31):
        #     z = trn.CenterCrop(64)(trn_F.affine(img, angle=0, translate=(0, 0),
        #                                          scale=1., shear=i-15, resample=PILImage.BILINEAR))
        #     z.save(os.path.join(tmp, 'img' + str(i) + '.png'))
        #
        # save_dir = '/share/data/vision-greg2/users/dan/datasets/ImageNet-64x64-P/shear/' + folder + '/'
        #
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        #
        # os.system("./ffmpeg/ffmpeg -r 1 -framerate 1 -i " +
        #           os.path.join(tmp, 'img') + "%01d.png -vcodec libx264 -crf 20 -preset veryslow -y " +
        #           save_dir + img_loc[:-5] + ".mp4")
        #
        # rmtree(tmp, ignore_errors=True)
        #
        # # /////////////// End Shear Code ///////////////
