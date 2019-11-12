import torch
import torch.nn.functional as F
#from libs import InPlaceABN, InPlaceABNSync
from torch import nn
from torch.nn import init
import math


class _Unary(nn.Module):

    def __init__(self, dim, inplanes, planes, downsample, use_gn, lr_mult, use_out, out_bn):
        assert dim in [1, 2, 3], "dim {} is not supported yet".format(dim)
        # assert whiten_type in ['channel', 'spatial']
        if dim == 3:
            conv_nd = nn.Conv3d
            if downsample:
                max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm3d
        elif dim == 2:
            conv_nd = nn.Conv2d
            if downsample:
                max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            if downsample:
                max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
            else:
                max_pool = None
            bn_nd = nn.BatchNorm1d

        super(_Unary, self).__init__()
        if use_out:
            self.conv_value = conv_nd(inplanes, planes, kernel_size=1)
            self.conv_out = conv_nd(planes, inplanes, kernel_size=1, bias=False)
        else:
            self.conv_value = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
            self.conv_out = None
        if out_bn:
            self.out_bn = nn.BatchNorm2d(inplanes)
        else:
            self.out_bn = None
        self.conv_mask = conv_nd(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.downsample = max_pool
        # self.norm = nn.GroupNorm(num_groups=32, num_channels=inplanes) if use_gn else InPlaceABNSync(num_features=inplanes)
        self.scale = math.sqrt(planes)

        self.reset_parameters()
        self.reset_lr_mult(lr_mult)

    def reset_parameters(self):

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
                m.inited = True
        # init.constant_(self.norm.weight, 0)
        # init.constant_(self.norm.bias, 0)
        # self.norm.inited = True

    def reset_lr_mult(self, lr_mult):
        if lr_mult is not None:
            for m in self.modules():
                m.lr_mult = lr_mult
        else:
            print('not change lr_mult')

    def forward(self, x):
        # [N, C, T, H, W]
        residual = x
        # [N, C, T, H', W']
        if self.downsample is not None:
            input_x = self.downsample(x)
        else:
            input_x = x

        # [N, C', T, H', W']
        value = self.conv_value(input_x)
        # [N, C', H' x W']
        value = value.view(value.size(0), value.size(1), -1)

        # [N, 1, H', W']
        mask = self.conv_mask(input_x)
        # [N, 1, H'x W']
        mask = mask.view(mask.size(0), mask.size(1), -1)
        mask = self.softmax(mask)
        # [N, C', 1, 1]
        out = torch.bmm(value, mask.permute(0, 2, 1)).unsqueeze(-1)

        # [N, C, T,  H, W]
        if self.conv_out is not None:
            out = self.conv_out(out)
        if self.out_bn:
            out = self.out_bn(out)

        out = out + residual

        return out


class Unary2d(_Unary):

    def __init__(self, inplanes, planes, downsample=True, use_gn=False, lr_mult=None, use_out=False, out_bn=False):
        super(Unary2d, self).__init__(dim=2, inplanes=inplanes, planes=planes, downsample=downsample,
                                            use_gn=use_gn, lr_mult=lr_mult, use_out=use_out, out_bn=out_bn)
