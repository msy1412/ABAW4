import os, sys, shutil
import csv
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms



def load_imgs_with_FER(csv_label,img_folder_path,mode="discrete"):
    imgs = list()
    ABAW4_dict={'ANGRER':0, 'DISGUST':1, 'FEAR':2, 'HAPPINESS':3, 'SADNESS':4, 'SURPRISE':5}
    with open(csv_label)as in_f:
        r_in_csv = csv.reader(in_f)
        for i,row in enumerate(r_in_csv):
            if i == 0:
                continue
            img_name = row[0]
            FER_label= ABAW4_dict[row[1]] #int(row[1].replace(' ',''))-1
            confidence = float(row[2].replace(' ',''))
            if confidence<0.8:
                continue
            label_arr=[]
            for i in row[3:]:
                value=float(i.replace(' ',''))
                if mode=="discrete":
                    if value>0.5:
                        label_arr.append(str(1))
                    else:
                        label_arr.append(str(0))
                elif mode=="continuous":
                    # if value>1:
                    #     label_arr.append(str(1))
                    # else:
                    label_arr.append(str(value))
            img_path =os.path.join(img_folder_path,img_name)
            if os.path.exists(img_path):
                imgs.append((img_path,FER_label,label_arr))
    return imgs


class ABAW_FER_AU(data.Dataset):
    def __init__(self, csv_label,img_folder_path, transform=None,mode="discrete"):
        self.imgs= load_imgs_with_FER(csv_label,img_folder_path,mode=mode)
        self.transform = transform
    def __getitem__(self, index):
        path,FER_target, target = self.imgs[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img,FER_target,target
    
    def __len__(self):
        return len(self.imgs)
