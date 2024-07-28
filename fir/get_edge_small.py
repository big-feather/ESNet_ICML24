import cv2
import os
import numpy as np

from tqdm import tqdm
from PIL import Image

ws='mask/'
out='edge/'

if not os.path.exists(out):
    os.makedirs(out)
gts=[]
for root, dirs, files in os.walk(ws):
    gts=files

for g in tqdm(gts):
    a=cv2.imread(ws+g,0)
    shape=a.shape
    # print(shape)
    a=cv2.resize(a,(352,352))
    a_e=cv2.Canny(a,20, 250)
    # a_e_m=cv2.GaussianBlur(a_e, (3, 3), 0)
    # a_e_m[a_e_m>0]=255
    a_e_m=a_e
    a_e_m[a_e_m > 0] = 255
    a_e_m=cv2.resize(a_e_m,(shape[1],shape[0]))
    c=Image.fromarray(a_e_m)
    c.save(out+g.split('.')[0]+'.'+g.split('.')[1])
