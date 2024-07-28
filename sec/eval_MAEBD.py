import cv2
import os
import numpy as np
from  matplotlib import pyplot as plt
import PIL
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F


class whole_detail_eval:
    def __init__(self, h=16, w=16, th=0.25, th1=30, m='MAE', k_s=25, quick_mode=False):
        self.h = h
        self.w = w
        self.th = th
        self.th1 = th1
        self.m = m
        self.k_s = k_s
        if k_s % 2 == 0:
            print('check k_s!')
        self.quick_mode = quick_mode
        if quick_mode:
            self.k_s = 5
        self.p = nn.MaxPool2d((self.k_s, self.k_s), stride=1, padding=self.k_s // 2).cuda()

    def judge(self, mask, edge, pred):
        H, W = mask.shape
        local_mask = torch.zeros((1, 1, H, W)).cuda()
        whole_mask = torch.tensor(edge).view(1, 1, H, W).cuda()
        p_h, p_w = H // self.h, W // self.w
        cnt = 0
        for h in range(self.h):
            for w in range(self.w):
                cnt_e = np.sum(edge[h * p_h:h * p_h + p_h, w * p_w:w * p_w + p_w])
                if cnt_e > self.th1:
                    cnt_m = np.sum(mask[h * p_h:h * p_h + p_h, w * p_w:w * p_w + p_w])
                    if cnt_e / (cnt_m +1e-8)> self.th:
                        #                         print(cnt_e)
                        cnt += 1
                        local_mask[0, 0, h * p_h:h * p_h + p_h, w * p_w:w * p_w + p_w] = 1

        #         print(cnt,mae_sum/(cnt+1e-8))
        local_mask = torch.tensor(mask).view(1, 1, H, W).cuda() * local_mask
        if self.quick_mode:
            local_mask = F.interpolate(local_mask, size=(400, 400), mode='bilinear', align_corners=False)
            whole_mask = F.interpolate(whole_mask, size=(400, 400), mode='bilinear', align_corners=False)
            local_mask = self.p(local_mask)
            whole_mask = self.p(whole_mask)
            local_mask = F.interpolate(local_mask, size=(H, W), mode='bilinear', align_corners=False).cpu()
            whole_mask = F.interpolate(whole_mask, size=(H, W), mode='bilinear', align_corners=False).cpu()
        else:
            local_mask = self.p(local_mask).cpu()
            whole_mask = self.p(whole_mask).cpu()

        whole_mask[whole_mask > 0] = 1
        local_mask[local_mask > 0] = 1
        # mae

        mae = np.abs(mask - pred)
        #         print((whole_mask[0,0]>0).shape,mae.shape,whole_mask.shape)
        mae_whole = np.mean(mae[whole_mask[0, 0] > 0])
        mae_local = np.mean(mae[local_mask[0, 0] > 0])

        # iou
        mask_whole = mask * whole_mask[0, 0].numpy()
        mask_local = mask * local_mask[0, 0].numpy()

        pred_whole = pred * whole_mask[0, 0].numpy()
        pred_local = pred * local_mask[0, 0].numpy()

        c_whole = np.sum(mask_whole * pred_whole)
        u_whole = np.sum(mask_whole) + np.sum(pred_whole) - c_whole
        iou_whole = c_whole / (u_whole + 1e-8)

        c_local = np.sum(mask_local * pred_local)
        u_local = np.sum(mask_local) + np.sum(pred_local) - c_local
        iou_local = c_local / (u_local + 1e-8)

        return local_mask, whole_mask, mae_whole, iou_whole, mae_local, iou_local, cnt

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    p = './out_path/' #pred path

    models = os.listdir(p)
    print(models)

    data_name=''
    mask_path = '../data/test/UHRSD_TE_2K_mask/'
    edge_path = '../data/test/UHRSD_TE_2K_mask_edge/'


    save_path1 = '../data/test//uhd_mask_0.25_15_B/'
    save_path2 = '../data/test//uhd_mask_0.25_15_D/'
    # fir_1210ssim_60
    if not os.path.exists(save_path1):
        os.makedirs(save_path1)
        os.makedirs(save_path2)
    de = whole_detail_eval(th1=30, th=0.25, k_s=15, quick_mode=True)
    gts = []

    flag = 0
    mae_whole_l, iou_whole_l, mae_local_l, iou_local_l, w_mae_l, w_iou_l = [], [], [], [], [], []
    for model in models:
        torch.cuda.empty_cache()
        pred_path = p + model + '/UHRSD_TE_2K/'
        # pred_path=mask_path
        for root, dirs, files in os.walk(mask_path):
            gts = files
        print(model)
        mae_whole_l, iou_whole_l, mae_local_l, iou_local_l, w_mae_l, w_iou_l = [], [], [], [], [], []
        for g in tqdm(gts):
            # print(mask_path + g)
            mask = cv2.imread(mask_path + g, 0) / 255
            edge = cv2.imread(edge_path + g.split('.')[0]+'.'+g.split('.')[1], 0) / 255
            pred = cv2.imread(pred_path + g, 0) / 255
            # print(mask.shape)
            pred = cv2.resize(pred,(mask.shape[1],mask.shape[0]))
            local_mask, whole_mask, mae_whole, iou_whole, mae_local, iou_local, cnt = de.judge(mask, edge, pred)
            #         print(mae_whole,iou_whole,mae_local,iou_local,cnt)
            mae_whole_l.append(mae_whole)

            iou_whole_l.append(iou_whole)
            # if np.isnan(mae_local):
            #     print(mae_whole, iou_whole, mae_local, iou_local, cnt)
            if cnt != 0 and not np.isnan(mae_local):
                iou_local_l.append(iou_local)
                mae_local_l.append(mae_local)
                s1 = torch.sum(whole_mask)
                s2 = torch.sum(local_mask)
                apl=s1/(s1+s2+1e-8)
                w_iou_l.append(apl*iou_local+(1-apl)*iou_whole)
                w_mae_l.append(apl*mae_local+(1-apl)*mae_whole)

            if flag == 0:
                f = PIL.Image.fromarray((whole_mask[0, 0].numpy() * 255).astype('uint8'))
                f.save(save_path1 + g)
                f = PIL.Image.fromarray((local_mask[0, 0].numpy() * 255).astype('uint8'))
                f.save(save_path2 + g)
        flag = 1
        print(model, '   MAE_BD:', np.mean(w_mae_l))





