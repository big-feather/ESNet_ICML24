import os
gpu_id='7'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
import sys
sys.path.append('../')
from datetime import datetime
from tqdm import tqdm

from fir.ESNet_fir import *
from sec.model.ESNet_sec import sec_evo

from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from data import get_loader

from options_final import opt
from options_dut_final import opt as opt_dut


#set the device for training
print('USE GPU ',gpu_id)
print(torch.cuda.is_available())
cudnn.benchmark = True
torch.cuda.empty_cache()


cfg = Config(datapath='path', snapshot='../checkpoints/ESNet_fir.pth', mode='test')


lr_model = CTDNet(cfg).train(False)


import fir.pytorch_msssim as pytorch_msssim

m = pytorch_msssim.MSSSIM()
f_flag=1
#set the path
image_root = opt.rgb_root
gt_root = opt.gt_root
gtp_root = opt.gtp_root
save_path=opt.save_path

image_root_w = opt_dut.rgb_root
gt_root_w = opt_dut.gt_root
gtp_root_w = opt_dut.gtp_root
save_path_w = opt_dut.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(save_path_w):
    os.makedirs(save_path_w)


#load data
print('load data....')
train_loader_hr = get_loader(image_root, gt_root,gtp_root, batchsize=4, trainsize=opt.trainsize,hr=1280,need_name=True,need_edge=True)
train_loader = get_loader(image_root, gt_root,gtp_root, batchsize=16, trainsize=256)


train_loader_hr_w = get_loader(image_root_w, gt_root_w,gtp_root_w, batchsize=24, trainsize=256,hr=512,need_name=True,need_edge=True)
train_loader_w = get_loader(image_root_w, gt_root_w,gtp_root_w, batchsize=16, trainsize=256)



total_step = len(train_loader_hr)

logging.basicConfig(filename=save_path+'log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("HR-Train")
logging.info("Config")
logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(opt.epoch,opt.lr,opt.batchsize,opt.trainsize,opt.clip,opt.decay_rate,opt.load,save_path,opt.decay_epoch))
print('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(opt.epoch,opt.lr,opt.batchsize,opt.trainsize,opt.clip,opt.decay_rate,opt.load,save_path,opt.decay_epoch))
#set loss function
CE = torch.nn.BCEWithLogitsLoss()
L1 = loss = nn.L1Loss()


step=0
writer = SummaryWriter(save_path+'summary')
best_mae=1
best_epoch=0

def total_loss(pred, mask):
    pred = torch.sigmoid(pred)
    bce_loss = nn.BCELoss()
    bce = bce_loss(pred, mask)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter+1)/(union-inter+1)
    iou = iou.mean()
    return iou , bce

def batch_norm(sn):
    s = sn.sigmoid()
    B = s.shape[0]
    for b in range(B):
        ma = torch.max(s[b,:,:,:])
        mi = torch.min(s[b,:,:,:])
        s[b,:,:,:] = (s[b,:,:,:] - mi) / (ma - mi + 1e-8)
    return s




def esnet_train(train_loader, lr_model, sec_model, optimizer, epoch,save_path, resolution = 1280, total_epoch=60):
    global step
    lr_model=lr_model.cuda()
    lr_model.eval()
    sec_model=sec_model.train().cuda()
    loss_all = 0
    epoch_step = 0
    pool = nn.MaxPool2d((7, 7), stride=1, padding=3).cuda()

    try:
        for i, (images, gts, gtps,images_hr,gts_hr,name) in enumerate(train_loader, start=1):

            optimizer.zero_grad()
            images_hr = images_hr.cuda()
            gts_hr = gts_hr.cuda()
            gtps=gtps.cuda()

            with torch.no_grad():
                images = F.interpolate(images_hr, size=images.shape[-2:], mode='bilinear', align_corners=False)
                s2, s1,l1,l2,path12,path1_2= lr_model(images)


            s_map = batch_norm(s2)
            s_map_1 = pool(s_map) - s_map
            s_map_2 = s_map - (-1 * pool(-1 * s_map))

            s_map = nn.Upsample(size=(resolution, resolution), mode='bilinear', align_corners=True)(s_map).detach()
            s_map_1 = nn.Upsample(size=(resolution, resolution), mode='bilinear', align_corners=True)(s_map_1).detach()
            s_map_2 = nn.Upsample(size=(resolution, resolution), mode='bilinear', align_corners=True)(s_map_2).detach()

            s_map_1 = batch_norm(s_map_1)
            s_map_2 = batch_norm(s_map_2)

            logits_1,  logits_edge, logits_diff, logits_diff1 = sec_model(images_hr, path12, path1_2, l1, l2,
                                                                                   s_map_1, s_map_2)
            l_s=torch.sigmoid(logits_1)
            s_diff=gts_hr-s_map
            s_diff[s_diff<0]=0
            s_diff1 =s_map - gts_hr
            s_diff1[s_diff1 < 0] = 0

            loss_diff = CE(logits_diff, s_diff)
            loss_diff1 = CE(logits_diff1, s_diff1)
            loss_main1,loss_main2 = total_loss(logits_1,gts_hr)
            loss_main_e = CE(logits_edge, gtps)
            ssim=1-m(l_s,gts_hr)
            loss_main = loss_main1 + 0.6* loss_main2 + loss_main_e + ssim*0.5
            loss_total = (loss_diff+loss_diff1)/5+ loss_main

            loss_total.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if step%40==0 :
                print('Loss_ssim: {:.4f} Loss_bce2: {:.4f} Loss_diff: {:.4f} Loss_main: {:.4f} Loss_total: {:.4f}'
                      ''.format(ssim.data,loss_main2.data, loss_diff.data, loss_main.data, loss_total.data))


            step += 1
            epoch_step += 1

            if (i % 80 == 0 or i == total_step or i == 1) and  f_flag > 0:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}]\n'
                      'Loss_ssim: {:.4f} Loss_bce2: {:.4f} Loss_diff: {:.4f} Loss_main: {:.4f} Loss_total: {:.4f}'.
                      format(datetime.now(), epoch, total_epoch, i, total_step, ssim.data,loss_main2.data,
                             loss_diff.data,loss_main.data,loss_total.data))

                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}]\n'
                      'Loss_ssim: {:.4f} Loss_bce2: {:.4f} Loss_diff: {:.4f} Loss_main: {:.4f} Loss_total: {:.4f}'.
                             format(epoch, total_epoch, i, total_step, ssim.data,loss_main2.data,
                                    loss_diff.data,loss_main.data,loss_total.data))


        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, total_epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if  epoch % 3 == 0:
            torch.save(sec_model.state_dict(), save_path + 'epoch_{}.pth'.format(epoch))
            # torch.save(p_model.state_dict(), save_path_p + 'epoch_{}.pth'.format(epoch))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(sec_model.state_dict(), save_path + 'epoch_{}.pth'.format(epoch+1))
        print('save  successfully!')
        raise

 
if __name__ == '__main__':
    print("Start train...")

    sec_model = sec_evo()


    optimizer_s = torch.optim.Adam(sec_model.parameters(), opt_dut.lr)

    lr_model.cuda()



    for epoch in range(1, opt_dut.epoch):
        cur_lr=adjust_lr(optimizer_s, opt_dut.lr, epoch, opt_dut.decay_rate, opt_dut.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        esnet_train(train_loader_hr_w, lr_model,sec_model, optimizer_s, epoch, save_path_w, resolution=512, total_epoch=opt_dut.epoch)


    optimizer_s = torch.optim.Adam(sec_model.parameters(), opt.lr)
    for epoch in range(1, opt.epoch):
        cur_lr=adjust_lr(optimizer_s, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)

        esnet_train(train_loader_hr, lr_model, sec_model, optimizer_s, epoch, save_path, total_epoch=opt.epoch)
    print('over!')


