import torch
import torch.nn as nn
import numpy as np
class multi_label6(nn.Module):
    def __init__(self):
        super(multi_label6, self).__init__()
        self.p1=nn.MaxPool2d((7,7),stride=2,padding=3)
        self.p2 = nn.MaxPool2d((3, 3), stride=2, padding=1)
    def forward(self,x):
        x_1=self.p1(x)
        x_2=self.p2(x_1)
        x_3 = self.p2(x_2)
        x_4 = self.p2(x_3)
        x_5 = self.p2(x_4)

        return x, x_1, x_2, x_3, x_4,x_5

class multi_label(nn.Module):
    def __init__(self):
        super(multi_label, self).__init__()
        self.p1=nn.MaxPool2d((7,7),stride=2,padding=3)
        self.p2 = nn.MaxPool2d((3, 3), stride=2, padding=1)
    def forward(self,x):
        x_1=self.p1(x)
        x_2=self.p2(x_1)
        x_3 = self.p2(x_2)
        x_4 = self.p2(x_3)

        return x, x_1, x_2, x_3, x_4

class multi_label_mean(nn.Module):
    def __init__(self):
        super(multi_label_mean, self).__init__()
        self.p1=nn.AvgPool2d((7,7),stride=2,padding=3)
        self.p2 = nn.AvgPool2d((3, 3), stride=2, padding=1)
    def forward(self,x):
        x_1=self.p1(x)
        x_2=self.p2(x_1)
        x_3 = self.p2(x_2)
        x_4 = self.p2(x_3)

        return x, x_1, x_2, x_3, x_4

class multi_label_gauss(nn.Module):
    def __init__(self):
        super(multi_label_gauss, self).__init__()
        self.p1=get_gaussian_kernel(7)
        self.p2 = get_gaussian_kernel(3)
        self.d=self.upsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
    def forward(self,x):
        x_1=self.d(self.p1(x))
        x_2=self.d(self.p2(x_1))
        x_3 = self.d(self.p2(x_2))
        x_4 = self.d(self.p2(x_3))

        return x, x_1, x_2, x_3, x_4



import torch
import math
import torch.nn as nn


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter
def custom_sobel(shape, axis):
    """
    shape must be odd: eg. (5,5)
    axis is the direction, with 0 to positive x and 1 to positive y
    """
    k = np.zeros(shape)
    p = [(j,i) for j in range(shape[0])
           for i in range(shape[1])
           if not (i == (shape[1] -1)/2. and j == (shape[0] -1)/2.)]

    for j, i in p:
        j_ = int(j - (shape[0] -1)/2.)
        i_ = int(i - (shape[1] -1)/2.)
        k[j,i] = (i_ if axis==0 else j_)/float(i_*i_ + j_*j_)
    return k

def get_sobel_kernels(channels=1,kernel_size=3,ax=0):
    sobel_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                             bias=False, padding=kernel_size // 2)
    snp = custom_sobel((kernel_size,kernel_size),ax).astype(np.float32)
    sobel_kernel = torch.from_numpy(snp)
    sobel_kernel = sobel_kernel.view(1, 1, kernel_size, kernel_size)
    sobel_kernel = sobel_kernel.repeat(channels, 1, 1, 1)
    sobel_filter.weight.data = sobel_kernel
    sobel_filter.weight.requires_grad = False
    return sobel_filter


def get_sobel_kernel(channels=1):
    kernel_size=3
    sobel_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)
    snp=np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    sobel_kernel = torch.from_numpy(snp)
    sobel_kernel = sobel_kernel.view(1, 1, kernel_size, kernel_size)
    sobel_kernel = sobel_kernel.repeat(channels, 1, 1, 1)
    sobel_filter.weight.data = sobel_kernel
    sobel_filter.weight.requires_grad = False
    return sobel_filter

def get_sobel_kernel_5(channels=1):
    kernel_size=3
    sobel_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)
    snp=np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    sobel_kernel = torch.from_numpy(snp)
    sobel_kernel = sobel_kernel.view(1, 1, kernel_size, kernel_size)
    sobel_kernel = sobel_kernel.repeat(channels, 1, 1, 1)
    sobel_filter.weight.data = sobel_kernel
    sobel_filter.weight.requires_grad = False
    return sobel_filter






