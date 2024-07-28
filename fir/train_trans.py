# coding=utf-8

import sys
import datetime
import random
import numpy as np
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset_swin as dataset
from ESNet_fir_swin import CTDNet
import os
from multi_label import *
# from apex import amp
import pytorch_msssim
from lr_scheduler import LR_Scheduler

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = pytorch_msssim.MSSSIM()
def total_loss_s(pred, mask):
    pred = torch.sigmoid(pred)
    bce_loss = nn.BCELoss()
    bce = bce_loss(pred, mask)
    ssim=1-m(pred,mask)
    # print(torch.max(pred),torch.min(pred),torch.max(mask),torch.min(mask))
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter+1)/(union-inter+1)
    iou = iou.mean()
    if torch.isnan(ssim):
        print('nan!')
        return iou + 0.6 * bce
    else:
        return iou + 0.6 * bce +ssim

def total_loss(pred, mask):
    pred = torch.sigmoid(pred)
    bce_loss = nn.BCELoss()
    bce = bce_loss(pred, mask)

    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter+1)/(union-inter+1)
    iou = iou.mean()
    return iou + 0.6*bce


def bce_loss(pred, mask):
    pred = torch.sigmoid(pred)
    bce_loss = nn.BCELoss()
    bce = bce_loss(pred, mask)
    return bce


def validate(model, val_loader, nums):
    model.train(False)
    avg_mae = 0.0
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image, mask = image.cuda().float(), mask.cuda().float()
            out, _, _, _, _, _ = model(image)
            pred = torch.sigmoid(out[0, 0])
            avg_mae += torch.abs(pred - mask[0]).mean()

    model.train(True)
    return (avg_mae / nums).item()


def train(Dataset, Network):
    ## Set random seeds
    seed = 7
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    ## dataset
    cfg = Dataset.Config(datapath='../data/train/HR_DUT/', savepath='./ckpt_fir_swin', mode='train', batch=18, lr=0.05, momen=0.9, decay=1e-4, epoch=60)
    if not os.path.exists(cfg.savepath):
        os.makedirs(cfg.savepath)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=4)

    multi=multi_label6()
    multi.cuda()
    up2= nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True).cuda()
    up0_25 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True).cuda()
    net = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    enc_params, dec_params = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            enc_params.append(param)
        else:
            dec_params.append(param)

    optimizer = torch.optim.SGD([{'params': enc_params}, {'params': dec_params}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    sw = SummaryWriter(cfg.savepath)
    global_step = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr
        for step, (image, mask, edge) in enumerate(loader):
            image, mask, edge = image.cuda().float(), mask.cuda().float(), edge.float().cuda()
            gt,gt1,gt2,gt3,gt4,gt5= multi(mask)
            optimizer.zero_grad()

            out1, out_edge, out2, out3, out4, out5,revo0,revo1,revo2,revo3 = net(image)

            loss_revo0 = bce_loss(revo0, up2(gt5) - gt4)
            loss_revo1 = bce_loss(revo1, up2(gt4)-gt3)
            loss_revo2 = bce_loss(revo2, up2(gt3)-gt2)
            loss_revo3 = bce_loss(revo3, gt2-up0_25(gt))
            loss1 = total_loss_s(out1, mask)


            loss_edge = bce_loss(out_edge, edge)
            loss2 = total_loss(out2, gt2)
            loss3 = total_loss(out3, gt3)
            loss4 = total_loss(out4, gt4)
            loss5 = total_loss(out5, gt5)
            loss = loss1 + loss_edge + (loss2/2 + loss3/4 + loss4/8 + loss5/16)+ (loss_revo0+ loss_revo1+ loss_revo2 + loss_revo3)/10
            # loss = loss1 + loss_edge + (loss2+ loss3 + loss4 + loss5)/10 + loss_revo0 + loss_revo1 + loss_revo2 + loss_revo3


            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scale_loss:
            #     scale_loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss1': loss1.item(), 'loss_edge': loss_edge.item(), 'loss2': loss2.item(),
                                    'loss3': loss3.item(), 'loss4': loss4.item(), 'loss5': loss5.item()}, global_step=global_step)
            if step % 10 == 0:
                print('%s | step:%d/%d/%d | lr=%.10f | loss=%.6f' % (datetime.datetime.now(), global_step, epoch+1,
                                                cfg.epoch, optimizer.param_groups[0]['lr'], loss.item()))

        if epoch > cfg.epoch-5 or epoch%5==0:
            torch.save(net.state_dict(), cfg.savepath + '/ESNet_fir_trans_' + str(epoch + 1)+'.pth')



if __name__ == '__main__':
    gpu_id='7'
    print('hhh')
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print('USE GPU ',gpu_id)
    train(dataset, CTDNet)
