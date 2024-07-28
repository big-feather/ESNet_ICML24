import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import argparse
import cv2
import sys
sys.path.append('./models')
sys.path.append('../')


from data import test_dataset


from fir.ESNet_fir import *
from model.ESNet_sec import sec_evo
torch.cuda.empty_cache()
import time

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=1280, help='testing size')
parser.add_argument('--gpu_id', type=str, default='7', help='select gpu id')
parser.add_argument('--test_path',type=str,default='../data/test/',help='test dataset path')

opt = parser.parse_args()

dataset_path = opt.test_path

cfg = Config(datapath='path', snapshot='../checkpoints/ESNet_fir.pth', mode='test') # path to the checkpoint

lr_model = CTDNet(cfg).train(False).cuda()

sec_model=sec_evo()

print(torch.cuda.is_available())
sec_model=sec_model.cuda().eval()

sec_model.load_state_dict(torch.load('../checkpoints/ESNet_sec.pth'))

def batch_norm(sn):
    s = sn.sigmoid()
    B = s.shape[0]
    for b in range(B):
        ma = torch.max(s[b,:,:,:])
        mi = torch.min(s[b,:,:,:])
        s[b,:,:,:] = (s[b,:,:,:] - mi) / (ma - mi + 1e-8)
    return s

lr_model.eval()


test_datasets = [ 'HRSOD_test', 'UHRSD_TE_2K']

for dataset in test_datasets:
    save_path = 'out_path/ESNet_60/' + dataset + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    image_root = dataset_path + dataset + '/'
    gt_root = dataset_path + dataset + '_mask/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    cost_time = list()
    mae=[]
    for i in range(test_loader.size):
        image_lr,image, gt, name, image_for_post,gt1 = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        image_lr = image_lr.cuda()
        pool = nn.MaxPool2d((7, 7), stride=1, padding=3).cuda()
        # depth = depth.cuda()
        with torch.no_grad():
            start_time = time.perf_counter()
            image_lr = F.interpolate(image, size=image_lr.shape[-2:], mode='bilinear', align_corners=False)
            s, _, l1, l2, path12, path1_2 = lr_model(image_lr)


            s_map = batch_norm(s)
            s_map_1 = pool(s_map) - s_map
            s_map_2 = s_map - (-1 * pool(-1 * s_map))

            s_map_1 = nn.Upsample(size=(1280, 1280), mode='bilinear', align_corners=True)(s_map_1).detach()
            s_map_2 = nn.Upsample(size=(1280, 1280), mode='bilinear', align_corners=True)(s_map_2).detach()

            s_map_1 = batch_norm(s_map_1)
            s_map_2 = batch_norm(s_map_2)
            logits_1,  logits_edge, _, _ = sec_model(image, path12, path1_2, l1, l2, s_map_1, s_map_2)
            cost_time.append(time.perf_counter() - start_time)

        res = F.interpolate(logits_1, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        print('save img to: ',save_path+name)
        cv2.imwrite(save_path + name, res * 255)

    print('Test Done!')
    cost_time.pop(0)
    print('Mean running time is: ', np.mean(cost_time))
    print("FPS is: ", test_loader.size / np.sum(cost_time))
exit(-1)