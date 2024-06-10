import os
import cv2
import ast
import pandas as pd
import torch, torchvision
from pytorchvideo.data.utils import thwc_to_cthw
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import v2

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings 

warnings.filterwarnings("ignore")


class LAVIBDataset(Dataset):
    def __init__(self, dataset_name, data_root='/media/SCRATCH/LAVIB', batch_size=32, set='main'):
        self.batch_size = batch_size
        self.dataset_name = dataset_name        
        self.h = 256
        self.w = 256
        self.dur = 3
        self.data_root = data_root
        
        if set=='main':
            addition=''
        elif set=='high_afm':
            addition='_high_fm'
        elif set=='low_afm':
            addition='_low_fm'
            
        elif set=='high_alv':
            addition='_high_lv'
        elif set=='low_alv':
            addition='_low_lv'
        
        elif set=='high_arl':
            addition='_high_pl'
        elif set=='low_arl':
            addition='_low_pl'
            
        elif set=='high_arms':
            addition='_high_rc'
        elif set=='low_arms':
            addition='_low_rc'
        
        else:
            addition=''
            
        self.addition = addition
        
        self.load_data()

    def __len__(self):
        return len(self.meta_data)
    
    def read_data(self,split):
        df = pd.read_csv(os.path.join(self.data_root,'annotations',split+f'{self.addition}.csv'))
        videos = [os.path.join(self.data_root,'segments_downsampled',f"{int(row['name'])}_shot{int(row['shot'])}_{int(row['tmp_crop'])}_{int(row['vrt_crop'])}_{int(row['hrz_crop'])}") for _,row in df.iterrows()]
           
        videos = sorted(videos)
        vidslist = []
        for vid in videos:
            for i in range(1,61-(self.dur*2),self.dur*2):
                vidslist.append([vid,(i,i+self.dur*2)])
        print(f'VideoLoader:: {len(vidslist)} frame tuples for {split} ready to be loaded with {self.dur} frames per tuple')
        
        return vidslist

    def load_data(self):
        self.meta_data = self.read_data(self.dataset_name)
        
    def crop(self, img0, gt, img1, h, w):
        _, ih, iw = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[:, x:x+h, y:y+w]
        img1 = img1[:, x:x+h, y:y+w]
        gt = gt[:, x:x+h, y:y+w]
        return img0, gt, img1

    def getimg(self, index):
        vidspath = self.meta_data[index]
        video_fr = torchvision.io.read_video(f"{vidspath[0]}/vid.mp4")[0]
        video_frames = [video_fr[i] for i in range(min(vidspath[1]),max(vidspath[1]),2)]

        # Load images
        video_frames = torch.stack(video_frames).float().permute(0, 3, 1, 2)
        if video_frames.shape[-1] > 720:
            if self.dataset_name != 'train':
                video_frames = v2.Resize(size=720,antialias=True)(video_frames)
        img0 = video_frames[0,...]
        img1 = video_frames[2,...]
        gt = video_frames[1,...]
        timestep = 0.5
        if self.dataset_name == 'train':
            img0, gt, img1 = self.crop(img0, gt, img1, self.h, self.w)
        return img0, gt, img1, timestep
            
    def __getitem__(self, index):        
        img0, gt, img1, timestep = self.getimg(index)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        return torch.cat((img0, img1, gt), 0), timestep




class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name        
        self.h = 256
        self.w = 448
        self.data_root = 'vimeo_triplet'
        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()   
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        cnt = int(len(self.trainlist) * 0.95)
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]
           
    def crop(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']

        # Load images
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        timestep = 0.5
        return img0, gt, img1, timestep
    
        # RIFEm with Vimeo-Septuplet
        # imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png', imgpath + '/im4.png', imgpath + '/im5.png', imgpath + '/im6.png', imgpath + '/im7.png']
        # ind = [0, 1, 2, 3, 4, 5, 6]
        # random.shuffle(ind)
        # ind = ind[:3]
        # ind.sort()
        # img0 = cv2.imread(imgpaths[ind[0]])
        # gt = cv2.imread(imgpaths[ind[1]])
        # img1 = cv2.imread(imgpaths[ind[2]])        
        # timestep = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0] + 1e-6)
            
    def __getitem__(self, index):        
        img0, gt, img1, timestep = self.getimg(index)
        if self.dataset_name == 'train':
            img0, gt, img1 = self.crop(img0, gt, img1, 224, 224)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                timestep = 1 - timestep
            # random rotation
            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        return torch.cat((img0, img1, gt), 0), timestep
    
