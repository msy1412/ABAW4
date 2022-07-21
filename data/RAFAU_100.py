import os, sys, shutil
import random as rd

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import transforms

def load_imgs(txt_label,img_folder_path,insert_index):
    imgs = list()
    with open(txt_label, 'r') as imf:
        for line in imf:
            line = line.strip()
            line = line.split()
            img_name = line[0]
            label_arr = line[1:]
            imgName_=list(img_name)
            imgName_.insert(insert_index,'_aligned')
            
            img_name="".join(imgName_)
            if img_folder_path.endswith("/") is False :
                img_folder_path+="/"
            img_path = img_folder_path + img_name
            imgs.append((img_path,label_arr))
    return imgs

class load_RAFAU(data.Dataset):
    def __init__(self, txt_label,img_folder_path, transform=None,phase=None):
        if phase=="train":
            insert_index=4
        elif phase=="test":
            insert_index=4
        self.imgs= load_imgs(txt_label,img_folder_path,insert_index)
        self.transform = transform
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.imgs)
