# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from fir.multi_label import get_gaussian_kernel
from fir.swin_encoder import SwinTransformer


def weight_init(module):
    for n, m in module.named_children():
        # print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.Upsample):
            pass
        else:
            m.initialize()


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            x = self.downsample(x)

        return F.relu(out+x, inplace=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            x = self.downsample(x)

        return F.relu(out+x, inplace=True)


# Feature Fusion Module
class FFM(nn.Module):
    def __init__(self, channel):
        super(FFM, self).__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_2):
        out = x_1 * x_2
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


# Cross Aggregation Module
class CAM(nn.Module):
    def __init__(self, channel):
        super(CAM, self).__init__()
        self.down = nn.Sequential(
            conv3x3(channel, channel, stride=2),
            nn.BatchNorm2d(channel)
        )
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.mul = FFM(channel)

    def forward(self, x_high, x_low):
        left_1 = x_low
        left_2 = F.relu(self.down(x_low), inplace=True)
        right_1 = F.interpolate(x_high, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        right_2 = x_high
        left = F.relu(self.bn_1(self.conv_1(left_1 * right_1)), inplace=True)
        right = F.relu(self.bn_2(self.conv_2(left_2 * right_2)), inplace=True)
        # left = F.relu(left_1 * right_1, inplace=True)
        # right = F.relu(left_2 * right_2, inplace=True)
        right = F.interpolate(right, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        out = self.mul(left, right)
        return out

    def initialize(self):
        weight_init(self)


# Spatial Attention Module
class SAM(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(SAM, self).__init__()
        self.conv_atten = conv3x3(2, 1)
        self.conv = conv3x3(in_chan, out_chan)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        atten = torch.cat([avg_out, max_out], dim=1)
        atten = torch.sigmoid(self.conv_atten(atten))
        out = torch.mul(x, atten)
        out = F.relu(self.bn(self.conv(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


# Boundary Refinement Module
class BRM(nn.Module):
    def __init__(self, channel):
        super(BRM, self).__init__()
        self.conv_atten = conv1x1(channel, channel)
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_edge):
        # x = torch.cat([x_1, x_edge], dim=1)
        x = x_1 + x_edge
        atten = F.avg_pool2d(x, x.size()[2:])
        atten = torch.sigmoid(self.conv_atten(atten))
        out = torch.mul(x, atten) + x
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)

class revo(nn.Module):
    def __init__(self, k=3,sig=2):
        super(revo, self).__init__()
        self.guass=get_gaussian_kernel(k,sig)
    def forward(self,f,s):
        se=s-torch.mul(s,s-self.guass(s))
        r=torch.mul(f+torch.mul(f,se),s)+f
        return r
    def initialize(self):
        weight_init(self)

class diff(nn.Module):
    def __init__(self,c1,c2,cn=1,up=True):
        super(diff, self).__init__()
        self.conv_1 = conv3x3(c1, cn)
        self.bn_1 = nn.BatchNorm2d(cn)
        self.conv_2 = conv3x3(c2, cn)
        self.bn_2 = nn.BatchNorm2d(cn)
        self.up2=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out = conv3x3(cn, 1, bias=True)
        self.up=up
    def forward(self, f1,f2,s):
        f11 = self.conv_1(f1)
        f11 = self.bn_1(f11)
        f22 = self.conv_2(f2)
        f22 = self.bn_2(f22)
        if self.up:
            f22=self.up2(f22)
            s=self.up2(s)
        f11=torch.mul(f11,s)
        f22 = torch.mul(f22, s)
        o = self.out(f22-torch.mul(f22,f11))
        return o
    def initialize(self):
        weight_init(self)



        
class CTDNet(nn.Module):
    def __init__(self, cfg):
        super(CTDNet, self).__init__()
        self.cfg = cfg
        block = BasicBlock
        # self.bkbone = ResNet(block, [2, 2, 2, 2])
        # self.bkbone = ResNet1()
        self.bkbone= SwinTransformer(img_size=384,
                                           embed_dim=128,
                                           depths=[2,2,18,2],
                                           num_heads=[4,8,16,32],
                                           window_size=12)


        self.path1_1 = nn.Sequential(
            conv1x1(1024 * block.expansion, 64),
            nn.BatchNorm2d(64)
        )
        self.path1_2 = nn.Sequential(
            conv1x1(1024 * block.expansion, 64),
            nn.BatchNorm2d(64)
        )
        self.path1_3 = nn.Sequential(
            conv1x1(512 * block.expansion, 64),
            nn.BatchNorm2d(64)
        )

        self.path2 = SAM(256 * block.expansion, 64)

        self.path3 = nn.Sequential(
            conv1x1(128 * block.expansion, 64),
            nn.BatchNorm2d(64)
        )

        self.fuse1_1 = FFM(64)
        self.fuse1_2 = FFM(64)
        self.fuse12 = CAM(64)
        self.fuse3 = FFM(64)
        self.fuse23 = BRM(64)

        self.head_1 = conv3x3(64, 1, bias=True)
        self.head_2 = conv3x3(64, 1, bias=True)
        self.head_3 = conv3x3(64, 1, bias=True)
        self.head_4 = conv3x3(64, 1, bias=True)
        self.head_5 = conv3x3(64, 1, bias=True)
        self.head_edge = conv3x3(64, 1, bias=True)

        self.r0 = revo(3,1)
        self.r1=  revo(3,1)
        self.r2 = revo(5,2)
        self.r3 = revo(9,4)
        self.r4 = revo(13,7)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.d0 = diff(64, 64)
        self.d1 = diff(64,64)
        self.d2 = diff(64, 64)
        self.d3 = diff(64, 64,up=False)



        self.initialize()
        if cfg.mode=='train':
            pretrained_dict = torch.load('../pretrained/swin_base_patch4_window12_384_22k.pth')["model"]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.bkbone.state_dict()}
            self.bkbone.load_state_dict(pretrained_dict)




    def forward(self, x, shape=None):
        shape = x.size()[2:] if shape is None else shape

        _,l5,l4,l3,l2,= self.bkbone(x)

        path1_1 = F.avg_pool2d(l5, l5.size()[2:])
        # print(path1_1.shape)
        path1_1 = self.path1_1(path1_1)
        path1_1 = F.interpolate(path1_1, size=l5.size()[2:], mode='bilinear', align_corners=True)   # 1/32
        s5=self.head_5(path1_1)

        path1_2 = F.relu(self.path1_2(l5), inplace=True)                                            # 1/32
        tmp0 = self.fuse1_1(path1_1, path1_2)
        path1_2 = self.r0(tmp0,s5.sigmoid())                                                        # 1/32

        path1_2 = F.interpolate(path1_2, size=l4.size()[2:], mode='bilinear', align_corners=True)   # 1/16
        path1_3 = F.relu(self.path1_3(l4), inplace=True)  # 1/16
        # path1 = torch.mul(self.fuse1_2(path1_2, path1_3),s4.sigmoid())# 1/16

        tmp1 = self.fuse1_2(path1_2, path1_3)
        revo0 = self.d0(tmp1, tmp0,s5.sigmoid())
        s4 = self.head_4(path1_2)+self.up2(s5)-revo0

        path1 = self.r1(tmp1, s4.sigmoid())
        # path1 = F.interpolate(path1, size=l3.size()[2:], mode='bilinear', align_corners=True)

        path2 = self.path2(l3)
        # 1/8
        tmp2 = self.fuse12(path1, path2)
        revo1 = self.d1(tmp2, tmp1,s4.sigmoid())
        s3 = self.up2(self.head_3(path1))+self.up2(s4)-revo1
        # s3 = F.interpolate(s3, size=l3.size()[2:], mode='bilinear', align_corners=True)

        path12 = self.r2(tmp2,s3.sigmoid())                                  # 1/8
        path12 = F.interpolate(path12, size=l2.size()[2:], mode='bilinear', align_corners=True)     # 1/4

        path3_1 = F.relu(self.path3(l2), inplace=True)  # 1/4
        path3_2 = F.interpolate(path1_2, size=l2.size()[2:], mode='bilinear', align_corners=True)  # 1/4
        tmp3 = self.fuse3(path3_1, path3_2)
        revo2 = self.d2(tmp3, tmp2,s3.sigmoid())
        s2=self.head_2(path12)+self.up2(s3)-revo2

        path3 = self.r3(tmp3,s2.sigmoid())                                # 1/4
        tmp4 =self.fuse23(path12, path3)
        path_out = self.r4(tmp4,s2.sigmoid())                               # 1/4



        revo3 = self.d3(tmp4, tmp3,s2.sigmoid())

        logits_1 = F.interpolate(self.head_1(path_out), size=shape, mode='bilinear', align_corners=True)
        logits_edge = F.interpolate(self.head_edge(path3), size=shape, mode='bilinear', align_corners=True)

        if self.cfg.mode == 'train':

            return logits_1, logits_edge, s2, s3, s4, s5, revo0, revo1, revo2, revo3

        else:

            s2 = F.interpolate(s2, size=shape, mode='bilinear', align_corners=True)

            return logits_1, s2, l2, l2, path12, path1_2

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77,  55.97,  57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None