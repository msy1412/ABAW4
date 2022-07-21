import os, sys, shutil
import random as rd

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import transforms

def load_imgs(txt_label):
    imgs = list()
    with open(txt_label, 'r') as imf:
        next(imf)
        for line in imf:
            img_name = line.strip()
            imgs.append(img_name)
    return imgs

class load_ABAW4_test(data.Dataset):
    def __init__(self, txt_label,img_folder_path, transform=None):
        self.imgs= load_imgs(txt_label)
        self.folder=img_folder_path
        self.transform = transform
    def __getitem__(self, index):
        imgName = self.imgs[index]
        img_path = os.path.join(self.folder,imgName)
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img,imgName
    
    def __len__(self):
        return len(self.imgs)