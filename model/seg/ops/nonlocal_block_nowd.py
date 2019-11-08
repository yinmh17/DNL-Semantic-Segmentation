import torch
import torch.nn.functional as F
#from libs import InPlaceABN, InPlaceABNSync
from torch import nn
from torch.nn import init
import math

class _NonLocalNd_nowd(nn.Module):

    def __init__(self, dim, inplanes, planes, downsample, lr_mult, use_out, out_bn, whiten_type, weight_init_scale, with_gc, with_nl, eps, nowd):
        assert dim in [1, 2, 3], "dim {} is not supported yet".format(dim)
        #assert whiten_type in ['in', 'in_nostd', 'ln', 'ln_nostd', 'fln', 'fln_nostd'] # all without affine, in == channel whiten
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

        super(_NonLocalNd_nowd, self).__init__()

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

        if with_nl:
            self.conv_query = conv_nd(inplanes, planes, kernel_size=1)
            self.conv_key = conv_nd(inplanes, planes, kernel_size=1)
        if with_gc:
            self.conv_mask = conv_nd(inplanes, 1, kernel_size=1)
       
        self.softmax = nn.Softmax(dim=2)
        self.downsample = max_pool
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(planes)
        self.whiten_type = whiten_type
        self.weight_init_scale = weight_init_scale
        self.with_gc = with_gc
        self.with_nl = with_nl
        self.nowd = nowd
        self.eps = eps
        
        self.reset_parameters()
        self.reset_lr_mult(lr_mult)
        self.reset_weight_and_weight_decay()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
                m.inited = True

    def reset_lr_mult(self, lr_mult):
        if lr_mult is not None:
            for m in self.modules():
                m.lr_mult = lr_mult
        else:
            print('not change lr_mult')

    def reset_weight_and_weight_decay(self):
        if self.with_nl:
            init.normal_(self.conv_query.weight, 0, 0.01*self.weight_init_scale)
            init.normal_(self.conv_key.weight, 0, 0.01*self.weight_init_scale)
            if 'nl' in self.nowd:
                self.conv_query.weight.wd=0.0
                self.conv_query.bias.wd=0.0
                self.conv_key.weight.wd=0.0
                self.conv_key.bias.wd=0.0
        if self.with_gc and 'gc' in self.nowd:
            self.conv_mask.weight.wd=0.0
            self.conv_mask.bias.wd=0.0
        if 'value' in self.nowd:
            self.conv_value.weight.wd=0.0
            #self.conv_value.bias.wd=0.0

    def forward(self, x):
        # [N, C, T, H, W]
        residual = x
        # [N, C, T, H', W']
        if self.downsample is not None:
            input_x = self.downsample(x)
        else:
            input_x = x

        value = self.conv_value(input_x)
        value = value.view(value.size(0), value.size(1), -1)

        out_sim = None

        if self.with_nl:
            # [N, C', T, H, W]
            query = self.conv_query(x)
            # [N, C', T, H', W']
            key = self.conv_key(input_x)

            # [N, C', H x W]
            query = query.view(query.size(0), query.size(1), -1)
            # [N, C', H' x W']
            key = key.view(key.size(0), key.size(1), -1)

            if 'in_nostd' in self.whiten_type :
                key_mean = key.mean(2).unsqueeze(2)
                query_mean = query.mean(2).unsqueeze(2)
                key -= key_mean
                query -= query_mean
            elif 'in' in self.whiten_type :
                key_mean = key.mean(2).unsqueeze(2)
                query_mean = query.mean(2).unsqueeze(2)
                key -= key_mean
                query -= query_mean
                key_var = key.var(2).unsqueeze(2)
                query_var = query.var(2).unsqueeze(2)
                key = key / torch.sqrt(key_var + self.eps)
                query = query / torch.sqrt(query_var + self.eps)
            elif 'ln_nostd' in self.whiten_type :
                key_mean = key.view(key.shape[0],-1).mean(1).unsqueeze(1).unsqueeze(2)
                query_mean = query.view(query.shape[0],-1).mean(1).unsqueeze(1).unsqueeze(2)
                key -= key_mean
                query -= query_mean
            elif 'ln' in self.whiten_type:
                key_mean = key.view(key.shape[0],-1).mean(1).unsqueeze(1).unsqueeze(2)
                query_mean = query.view(query.shape[0],-1).mean(1).unsqueeze(1).unsqueeze(2)
                key -= key_mean
                query -= query_mean
                key_var = key.view(key.shape[0],-1).var(1).unsqueeze(1).unsqueeze(2)
                query_var = query.view(query.shape[0],-1).var(1).unsqueeze(1).unsqueeze(2)
                key = key / torch.sqrt(key_var + self.eps)
                query = query / torch.sqrt(query_var + self.eps)
            elif 'fln_nostd' in self.whiten_type :
                key_mean = key.view(1,-1).mean(1).unsqueeze(1).unsqueeze(2)
                query_mean = query.view(1,-1).mean(1).unsqueeze(1).unsqueeze(2)
                key -= key_mean
                query -= query_mean
            elif 'fln' in self.whiten_type:
                key_mean = key.view(1,-1).mean(1).unsqueeze(1).unsqueeze(2)
                query_mean = query.view(1,-1).mean(1).unsqueeze(1).unsqueeze(2)
                key -= key_mean
                query -= query_mean
                key_var = key.view(1,-1).var(1).unsqueeze(1).unsqueeze(2)
                query_var = query.view(1,-1).var(1).unsqueeze(1).unsqueeze(2)
                key = key / torch.sqrt(key_var + self.eps)
                query = query / torch.sqrt(query_var + self.eps)

            # [N, T x H x W, T x H' x W']
            sim_map = torch.bmm(query.transpose(1, 2), key)
            ### cancel temp and scale
            if 'nl' not in self.nowd:
                sim_map = sim_map/self.scale
            sim_map = self.softmax(sim_map)

            # [N, T x H x W, C']
            out_sim = torch.bmm(sim_map, value.transpose(1, 2))
            # [N, C', T x H x W]
            out_sim = out_sim.transpose(1, 2)
            # [N, C', T,  H, W]
            out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
            out_sim = self.gamma * out_sim

        if self.with_gc:
            # [N, 1, H', W']
            mask = self.conv_mask(input_x)
            # [N, 1, H'x W']
            mask = mask.view(mask.size(0), mask.size(1), -1)
            mask = self.softmax(mask)
            # [N, C', 1, 1]
            out_gc = torch.bmm(value, mask.permute(0,2,1)).unsqueeze(-1)
            if out_sim is not None:
                out_sim = out_sim + out_gc
            else:
                out_sim = out_gc
            
        # [N, C, T,  H, W]
        if self.conv_out is not None:
            out_sim = self.conv_out(out_sim)
        if self.out_bn:
            out_sim = self.out_bn(out_sim)
            
        out = out_sim + residual

        return out


class NonLocal2d_nowd(_NonLocalNd_nowd):
    def __init__(self, inplanes, planes, downsample=True, lr_mult=None, use_out=False, out_bn=False, whiten_type=['in_nostd'], weight_init_scale=1.0, with_gc=False, with_nl=True, eps=1e-5, nowd=['nl']):
        super(NonLocal2d_nowd, self).__init__(dim=2, inplanes=inplanes, planes=planes, downsample=downsample, lr_mult=lr_mult, use_out=use_out, out_bn=out_bn, whiten_type=whiten_type, weight_init_scale=weight_init_scale, with_gc=with_gc, with_nl=with_nl, eps=eps, nowd=nowd)

