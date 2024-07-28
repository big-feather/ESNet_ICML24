# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sec.model.multi_label import get_gaussian_kernel,get_sobel_kernel
import math
from einops import rearrange, repeat


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
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


class Bottleneck1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=7, stride=stride,dilation=1, padding=3, bias=False)
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

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class decoder_end(nn.Module):
    def __init__(self,edge=False):
        super(decoder_end, self).__init__()
        block = BasicBlock
        self.fuse3 = FFM(64)
        self.fuse23 = BRM(64)
        self.path3 = nn.Sequential(
            conv1x1(128 * block.expansion, 64),
            nn.BatchNorm2d(64)
        )
        self.head_1 = conv3x3(64, 1, bias=True)

        self.edge=edge
        self.head_edge = conv3x3(64, 1, bias=True)
    def forward(self,path12,path1_2,l2,shape):
        path3_1 = F.relu(self.path3(l2), inplace=True)  # 1/4
        path3_2 = F.interpolate(path1_2, size=l2.size()[2:], mode='bilinear', align_corners=True)  # 1/4
        # print(path3_1.shape, path3_2.shape,'a')
        path3 = self.fuse3(path3_1, path3_2)  # 1/4

        path_out = self.fuse23(path12, path3)  # 1/4

        logits_1 = F.interpolate(self.head_1(path_out), size=shape, mode='bilinear', align_corners=True)
        if self.edge:
            logits_edge = F.interpolate(self.head_edge(path3), size=shape, mode='bilinear', align_corners=True)
            return logits_1,logits_edge,path_out
        return logits_1
    def initialize(self):
        weight_init(self)

class evo_encoder(nn.Module):
    def __init__(self):
        super(evo_encoder, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,dilation=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1)
        self.sfem=SFEM()
    def make_layer(self, planes, blocks, stride):
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck1(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck1(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self,x,m1,m2):
        l1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        l1 = F.max_pool2d(l1, kernel_size=7, stride=2, padding=3)
        l1 = self.sfem(l1,m1,m2)
        l2 = self.layer1(l1)
        l2 = self.sfem(l2, m1, m2)
        return l1,l2
    def initialize(self):
        weight_init(self)

class SFEM(nn.Module):
    def __init__(self):
        super(SFEM, self).__init__()
    def forward(self,f,m1,m2):
        B, C, H, W = f.shape
        v_s_sum = repeat(torch.sum(m1.reshape(B, H * W), dim=1, keepdim=True), 'b () -> b d', d=C)
        v_ns_sum = repeat(torch.sum(m2.reshape(B, H * W), dim=1, keepdim=True), 'b () -> b d', d=C)
        v_s = torch.sum(torch.mul(f, m1).reshape(B, C, H * W), dim=2) / (v_s_sum + 1e-8)
        v_ns = torch.sum(torch.mul(f, m2).reshape(B, C, H * W), dim=2) / (v_ns_sum + 1e-8)

        k = torch.cat([v_s.unsqueeze(-1), v_ns.unsqueeze(-1)], dim=-1)
        v = k.permute(0, 2, 1)

        f_m = f.permute(0, 2, 3, 1)
        q = f_m.reshape(B, H * W, C)
        norm_fact = 1 / math.sqrt(C)
        atten = nn.Softmax(dim=-1)(torch.bmm(q, k)) * norm_fact
        # print(atten.shape,v.shape)
        f_sa = torch.bmm(atten, v)
        f_sa = f_sa.reshape(B, H, W, C)
        # print(f.shape,f_sa.shape)
        return f + torch.mul(f_sa.permute(0, 3, 1, 2),m1+m2)

class sec_evo(nn.Module):
    def __init__(self):
        super(sec_evo, self).__init__()
        self.e1 = evo_encoder()

        self.d=decoder_end(True)
        self.d_diff = decoder_end()
        self.d_diff1 = decoder_end()

        self.so=get_sobel_kernel(256)
        self.conv=conv1x1(256,128)
        self.sfem=SFEM()


    def make_layer(self, planes, blocks, stride):
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck1(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck1(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self,x,path12,path1_2,l11,l22,mask,mask1):

        # l1_m, l2_m = self.e2(x)
        shape=(x.shape[-2]//4,x.shape[-1]//4)
        l2_hr = F.interpolate(l22, size=shape, mode='bilinear', align_corners=True)
        # l1_hr = F.interpolate(l11, size=l1_a.size()[2:], mode='bilinear', align_corners=True)

        path12 = F.interpolate(path12, size=shape, mode='bilinear', align_corners=True)
        path1_2= F.interpolate(path1_2, size=shape, mode='bilinear', align_corners=True)




        mask = F.interpolate(mask, size=shape, mode='bilinear', align_corners=False)
        mask1 = F.interpolate(mask1, size=shape, mode='bilinear', align_corners=False)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 1e-8
        mask1[mask1 > 0.5] = 1
        mask1[mask1 <= 0.5] = 1e-8
        l1_a, l2_a = self.e1(x, mask,mask1)
        f_en = l2_a

        f_en=l2_hr+l2_hr*(self.conv(self.so(f_en)).sigmoid())


        logits_1, logits_edge, f_1 = self.d(path12, path1_2, f_en, x.shape[2:])
        logits_diff = self.d_diff(path12, path1_2, l2_hr-f_en, x.shape[2:])
        logits_diff1 = self.d_diff1(path12, path1_2, f_en-l2_hr, x.shape[2:])
        return logits_1,logits_edge,logits_diff,logits_diff1

    def initialize(self):
        weight_init(self)
        
class MLP(nn.Module):
    def __init__(self,ind=128,hid=128):
        super(MLP, self).__init__()
        self.l1=nn.Sequential(
            nn.Linear(ind, hid),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU()
        )
        self.l4 = nn.Sequential(
            nn.Linear(hid, 1),
        )
    def forward(self,x,):
        x1=self.l1(x)
        x2=self.l2(x1)
        x3=self.l3(x2)
        x4=self.l4(x3)
        return x4

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

if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    print('USE GPU 4')

    print(torch.cuda.is_available())

    sec_model=sec_evo().cuda()
    optimizer = torch.optim.Adam(sec_model.parameters(), 0.001)
    CE = torch.nn.BCEWithLogitsLoss()


    while True:
        optimizer.zero_grad()
        dummy_x = torch.randn(1, 3, 1280, 1280).cuda()
        img = torch.randn(1, 3, 160, 160).cuda()
        dummy_Y = torch.randn(1, 1, 1280, 1280).cuda()

        e1,e2=sec_model(dummy_x,dummy_x,dummy_x,dummy_x,dummy_x)
        print(e1.shape,e2.shape)
        pass


