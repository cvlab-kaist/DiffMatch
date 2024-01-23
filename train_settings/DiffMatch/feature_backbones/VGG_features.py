from collections import OrderedDict

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def correlation(src_feat, trg_feat, eps=1e-5):
    src_feat = src_feat / (src_feat.norm(dim=1, p=2, keepdim=True) + eps)
    trg_feat = trg_feat / (trg_feat.norm(dim=1, p=2, keepdim=True) + eps)

    return th.einsum('bchw, bcxy -> bhwxy', src_feat, trg_feat)

class VGGPyramid(nn.Module):
    def __init__(self, train=False, pretrained=True):
        super().__init__()
        self.n_levels = 5
        source_model = models.vgg16(pretrained=pretrained)

        modules = OrderedDict()
        tmp = []
        n_block = 0
        first_relu = False

        for c in source_model.features.children():
            if (isinstance(c, nn.ReLU) and not first_relu) or (isinstance(c, nn.MaxPool2d)):
                first_relu = True
                tmp.append(c)
                modules['level_' + str(n_block)] = nn.Sequential(*tmp)
                for param in modules['level_' + str(n_block)].parameters():
                    param.requires_grad = train

                tmp = []
                n_block += 1
            else:
                tmp.append(c)

            if n_block == self.n_levels:
                break

        self.__dict__['_modules'] = modules

    def forward(self, x, quarter_resolution_only=False, eigth_resolution=False):
        outputs = []
        if quarter_resolution_only:
            x_full = self.__dict__['_modules']['level_' + str(0)](x)
            x_half = self.__dict__['_modules']['level_' + str(1)](x_full)
            x_quarter = self.__dict__['_modules']['level_' + str(2)](x_half)
            outputs.append(x_quarter)
        elif eigth_resolution:
            x_full = self.__dict__['_modules']['level_' + str(0)](x)
            outputs.append(x_full)
            x_half = self.__dict__['_modules']['level_' + str(1)](x_full)
            x_quarter = self.__dict__['_modules']['level_' + str(2)](x_half)
            outputs.append(x_quarter)
            x_eight = self.__dict__['_modules']['level_' + str(3)](x_quarter)
            outputs.append(x_eight)
        else:
            for layer_n in range(0, self.n_levels):
                x = self.__dict__['_modules']['level_' + str(layer_n)](x)
                outputs.append(x)
        return outputs
    
def extract_features(pyramid, im_target, im_source, im_target_256, im_source_256,
                         im_target_pyr=None, im_source_pyr=None, im_target_pyr_256=None, im_source_pyr_256=None):
        
        if im_target_pyr is None:
            im_target_pyr = pyramid(im_target, eigth_resolution=True)
        if im_source_pyr is None:
            im_source_pyr = pyramid(im_source, eigth_resolution=True)
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

        return c14, c24, c13, c23, c12, c22, c11, c21

@th.no_grad()
def extract_raw_features(pyramid, target, source, target_256, source_256, feature_size=64):
    c14, c24, c13, c23, c12, c22, c11, c21 = \
        extract_features(pyramid, target, source, target_256,
                    source_256, None, None, None, None)
    trg_feat = [c14, c13, c12, c11]
    src_feat = [c24, c23, c22, c21]
    trg_feat_list = []
    src_feat_list = []
    corr_list = []
    for trg, src in zip(trg_feat, src_feat):
        trg_feat_list.append(F.interpolate(trg, size=(feature_size), mode='bilinear', align_corners=False))
        src_feat_list.append(F.interpolate(src, size=(feature_size), mode='bilinear', align_corners=False))
    
    for trg, src in zip(trg_feat_list, src_feat_list):
        corr = correlation(src, trg)[:, None]
        corr_list.append(corr)

    raw_corr = sum(corr_list) / len(corr_list)

    c12 = F.interpolate(c12, size=(feature_size), mode='bilinear', align_corners=False)
    c22 = F.interpolate(c22, size=(feature_size), mode='bilinear', align_corners=False)
    return raw_corr, trg_feat, src_feat
    
if __name__ == "__main__":
    print("VGGPyramid test")
    model = VGGPyramid()
    print(model)
    breakpoint()
    print("Done.")