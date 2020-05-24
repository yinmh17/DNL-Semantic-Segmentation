#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Mobilenet models.

import torch.nn as nn
from model.backbone.mobilenet.mobilenet_models import MobileNetModels

class MobileNetBackbone(object):
    def __init__(self, configer):
        self.configer = configer
        self.mobile_models = MobileNetModels(self.configer)

    def __call__(self, backbone=None, pretrained=None):
        arch = self.configer.get('network.backbone') if backbone is None else backbone
        pretrained = self.configer.get('network.pretrained') if pretrained is None else pretrained

        if arch == 'mobilenetv2':
            arch_net = self.mobile_models.mobilenetv2(pretrained=pretrained)

        elif arch == 'mobilenetv2_dilated8':
            #arch_net = self.mobile_models.mobilenetv2_dilated8(pretrained=pretrained)
            orig_mobilenet = self.mobile_models.mobilenetv2(pretrained=pretrained)
            arch_net = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)

        else:
            raise Exception('Architecture undefined!')

        return arch_net


class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
#        print("==================================================")
#        print(self.features)
        self.features = nn.Sequential(*self.features)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    #m.dilation = (dilate//2, dilate//2)
                    #m.padding = (dilate//2, dilate//2)
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def get_num_features(self):
        return 320

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            f1 = self.features[:15](x)
            f2 = self.features[15:](f1)
            return [f1, f2]
