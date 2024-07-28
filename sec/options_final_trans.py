import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=61, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--lr_batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='7', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default='../data/train/HRSOD_train_all/', help='the training rgb images root')
parser.add_argument('--gt_root', type=str, default='../data/train/HRSOD_train_all_mask/', help='the training gt images root')
parser.add_argument('--gtp_root', type=str, default='../data/train/HR_DUT/edge//', help='the training gt images root')
parser.add_argument('--save_path', type=str, default='./esnet_swin/', help='the path to save models and logs')
opt = parser.parse_args()