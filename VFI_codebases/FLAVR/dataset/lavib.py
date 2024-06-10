import os
import numpy as np
import pandas as pd
import torch, torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from torchvision.transforms import v2

from pytorchvideo.data.utils import thwc_to_cthw
import warnings 

warnings.filterwarnings("ignore")

class LAVIB(Dataset):
    def __init__(self, data_root, is_training , set, input_frames="1357"):
        """
        Creates a Vimeo Septuplet object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            input_frames: Which frames to input for frame interpolation network.
        """
        self.data_root = data_root
        self.image_root = os.path.join(self.data_root, 'segments')
        self.training = is_training
        self.inputs = [int(i) for i in input_frames]
        self.dur = int(input_frames[-1]) - int(input_frames[0])+1

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
        
        if self.training:
            df = pd.read_csv(os.path.join(self.data_root,'annotations',f'train{addition}.csv'))
            self.split = 'train'
        else:
            df = pd.read_csv(os.path.join(self.data_root,'annotations',f'test{addition}.csv'))
            self.split = 'test'
        
        videos = [os.path.join(self.data_root,'segments',f"{int(row['name'])}_shot{int(row['shot'])}_{int(row['tmp_crop'])}_{int(row['vrt_crop'])}_{int(row['hrz_crop'])}") for _,row in df.iterrows()]
        
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
        if video_frames.shape[-1] > 720:
            if self.split != 'train':
                video_frames = v2.Resize(size=720,antialias=True)(video_frames)
        
        if self.split == 'train':
            video_frames = self.crop(video_frames, 256, 256)
        
        ## Select only relevant inputs
        inputs = [i-1 for i in self.inputs]
        inputs = inputs[:len(inputs)//2] + [3] + inputs[len(inputs)//2:]
        images = [video_frames[i] for i in inputs]
        
        gt = images[len(images)//2]
        images = images[:len(images)//2] + images[len(images)//2+1:]
        
        return images, gt

    def __len__(self):
        return len(self.vidslist)
        

def get_loader(mode, data_root, batch_size, shuffle, num_workers, set, test_mode=None):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = LAVIB(data_root, is_training=is_training, set=set)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":

    dataset = LAVIB("/media/SCRATCH/LAVIB", is_training=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    for dat in dataloader:
        print(f"inps shape {dat[0].shape} gt shape: {dat[1].shape}")