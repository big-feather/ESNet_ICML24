import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import cv2

#several data augumentation strategies
def cv_random_flip(img, label,label_p):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    #left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        label_p = label_p.transpose(Image.FLIP_LEFT_RIGHT)
    #top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label, label_p
def randomCrop(image, label, label_p):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), label_p.crop(random_region)
def randomRotation(image,label,label_p):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image=image.rotate(random_angle, mode)
        label=label.rotate(random_angle, mode)
        label_p = label_p.rotate(random_angle, mode)
    return image,label,label_p
def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))
def randomPeper(img):

    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):

        randX=random.randint(0,img.shape[0]-1)  

        randY=random.randint(0,img.shape[1]-1)  

        if random.randint(0,1)==0:  

            img[randX,randY]=0  

        else:  

            img[randX,randY]=255 
    return Image.fromarray(img)  

# dataset for training
#The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
#(e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, gtp_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.gtps=[gtp_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.gtps = sorted(self.gtps)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gtp_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        gt_p = self.binary_loader(self.gtps[index])
        image,gt,gt_p =cv_random_flip(image,gt,gt_p)
        image,gt,gt_p=randomCrop(image, gt , gt_p)
        image,gt,gt_p=randomRotation(image, gt, gt_p)
        image=colorEnhance(image)
        # gt=randomGaussian(gt)
        gt=randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        gtp = self.gtp_transform(gt_p)
        return image, gt,gtp

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []

        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
            else:
                print(img_path,gt_path)

        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

class SalObjDataset_hr(data.Dataset):
    def __init__(self, image_root, gt_root, gtp_root, trainsize, trainsize_hr,need_name=False,need_edge=False):
        self.trainsize_hr = trainsize_hr
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        # self.gtps = [gtp_root + f.split('.')[0] + '_edge.' + f.split('.')[1] for f in os.listdir(gt_root) if f.endswith
        self.gtps=[gtp_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.gtps = sorted(self.gtps)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.4884, 0.4663, 0.4037], [0.2226, 0.2195, 0.2255])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gtp_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.img_transform_hr = transforms.Compose([
            transforms.Resize((self.trainsize_hr, self.trainsize_hr)),
            transforms.ToTensor(),
            transforms.Normalize([0.4884, 0.4663, 0.4037], [0.2226, 0.2195, 0.2255])])
        self.gt_transform_hr = transforms.Compose([
            transforms.Resize((self.trainsize_hr, self.trainsize_hr)),
            transforms.ToTensor()])
        self.need_name=need_name
        self.need_edge = need_edge



    def __getitem__(self, index):

        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        gt_p = self.binary_loader(self.gtps[index])
        image,gt,gt_p =cv_random_flip(image,gt,gt_p)
        image,gt,gt_p=randomCrop(image, gt , gt_p)
        image,gt,gt_p=randomRotation(image, gt, gt_p)
        image=colorEnhance(image)
        # gt=randomGaussian(gt)


        image_hr=self.img_transform_hr(image)
        gt_hr = self.gt_transform_hr(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        gtp = self.gt_transform_hr(gt_p)
        if self.need_name:
            name = self.images[index].split('/')[-1]
            if self.need_edge:
                return image, gt, gtp, image_hr, gt_hr, str(name)
            else:
                return image, gt,gtp,image_hr,gt_hr,str(name)
        return image, gt,gtp,image_hr,gt_hr

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []

        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
            else:
                print(img_path,gt_path)

        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

#dataloader for training
def get_loader(image_root, gt_root, gtp_root, batchsize, trainsize, hr=0,shuffle=True, num_workers=16, pin_memory=True,need_name=False,need_edge=False):

    if hr>0:
        dataset = SalObjDataset_hr(image_root, gt_root, gtp_root, trainsize,hr,need_name=need_name,need_edge=need_edge)
    else:
        dataset = SalObjDataset(image_root, gt_root, gtp_root,trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

#test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root,  testsize,testsize_lr=352):
        self.testsize = testsize
        self.testsize_lr = testsize_lr
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.4884, 0.4663, 0.4037], [0.2226, 0.2195, 0.2255])])
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform_lr = transforms.Compose([
            transforms.Resize((self.testsize_lr, self.testsize_lr)),
            transforms.ToTensor(),
            transforms.Normalize([0.4884, 0.4663, 0.4037], [0.2226, 0.2195, 0.2255])])
        self.gt_transform = transforms.ToTensor()
        self.transform_gt1 = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()
        ])
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image_lr = self.transform_lr(image).unsqueeze(0)
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        gt1 = self.transform_gt1(gt).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        image_for_post=self.rgb_loader(self.images[self.index])
        image_for_post=self.transform_gt1(image_for_post).unsqueeze(0)
        # image_for_post=image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image_lr,image,gt,name,image_for_post,gt1

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size
class test_dataset1:
    def __init__(self, image_root, gt_root,  testsize,testsize_lr=256):
        self.testsize = testsize
        self.testsize_lr = testsize_lr
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform1 = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.transform_lr = transforms.Compose([
            transforms.Resize((self.testsize_lr, self.testsize_lr)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.transform_gt1 = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()
            ])
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image_lr = self.transform_lr(image).unsqueeze(0)
        image0 = self.transform1(image).unsqueeze(0)
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])

        name = self.images[self.index].split('/')[-1]
        image_for_post=self.rgb_loader(self.images[self.index])
        image_for_post=image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image_lr,image,gt,name,np.array(image_for_post),image0

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size

