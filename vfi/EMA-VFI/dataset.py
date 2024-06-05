import cv2
import os
import torch, torchvision
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset
from config import *
from torchvision.transforms import v2

from pytorchvideo.data.utils import thwc_to_cthw
import warnings 

warnings.filterwarnings("ignore")


cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LAVIBDataset(Dataset):
    def __init__(self, dataset_name, path, batch_size=32, model="RIFE", set='main'):
        """
        Creates a Vimeo Septuplet object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            input_frames: Which frames to input for frame interpolation network.
        """
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.model = model
        self.h = 256
        self.w = 256
        self.data_root = path
        self.image_root = os.path.join(self.data_root, 'segments_downsampled')
        
        self.dur = 3
        
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
            

        if self.dataset_name == 'train':
            df = pd.read_csv(os.path.join(self.data_root,'annotations',f'train{addition}.csv'))
            self.split = 'train'
        else:
            df = pd.read_csv(os.path.join(self.data_root,'annotations',f'test{addition}.csv'))
            self.split = 'test'
        
        videos = [os.path.join(self.data_root,'segments_downsampled',f"{int(row['name'])}_shot{int(row['shot'])}_{int(row['tmp_crop'])}_{int(row['vrt_crop'])}_{int(row['hrz_crop'])}") for _,row in df.iterrows()]
           
        videos = sorted(videos)
        self.vidslist = []
        for vid in videos:
            for i in range(1,61-(self.dur*2),self.dur*2):
                self.vidslist.append([vid,(i,i+self.dur*2)])
        print(f'VideoLoader:: {len(self.vidslist)} frame tuples for {self.split} ready to be loaded with {self.dur} frames per tuple')
    
    
    def crop(self, frames, h, w):
        _, _, ih, iw = frames.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        frames = frames[..., x:x+h, y:y+w]
        return frames      

    def __getitem__(self, index):
        
        vidspath = self.vidslist[index]
        video_fr = torchvision.io.read_video(f"{vidspath[0]}/vid.mp4")[0]
        #print(video_fr.shape, vidspath[1])
        video_frames = [video_fr[i] for i in range(min(vidspath[1]),max(vidspath[1]),2)]
        
        video_frames = torch.stack(video_frames).float().permute(0, 3, 1, 2)
        if video_frames.shape[-1] > 512:
            if self.split != 'train':
                video_frames = v2.Resize(size=512,antialias=True)(video_frames)
        
        if self.split == 'train':
            video_frames = self.crop(video_frames, 256, 256)
        
        return torch.cat((video_frames[0], video_frames[2], video_frames[1]), 0)

    def __len__(self):
        return len(self.vidslist)




class VimeoDataset(Dataset):
    def __init__(self, dataset_name, path, batch_size=32, model="RIFE"):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.model = model
        self.h = 256
        self.w = 448
        self.data_root = path
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
        if self.dataset_name != 'test':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist

    def aug(self, img0, gt, img1, h, w):
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
        
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        return img0, gt, img1
            
    def __getitem__(self, index):        
        img0, gt, img1 = self.getimg(index)
                
        if 'train' in self.dataset_name:
            img0, gt, img1 = self.aug(img0, gt, img1, 256, 256)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img1, img0 = img0, img1
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]

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
        return torch.cat((img0, img1, gt), 0)
