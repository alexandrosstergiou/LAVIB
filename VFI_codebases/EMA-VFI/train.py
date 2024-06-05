import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from tqdm import tqdm

from Trainer import Model
from dataset import LAVIBDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from config import *

device = torch.device("cuda")
exp = os.path.abspath('.').split('/')[-1]

def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024 ** 3

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000
        return 2e-4 * mul
    else:
        mul = np.cos((step - 2000) / (300 * args.step_per_epoch - 2000) * math.pi) * 0.5 + 0.5
        return (2e-4 - 2e-5) * mul + 2e-5

def train(model, local_rank, batch_size, data_path, tset):
    if local_rank == 0:
        writer = SummaryWriter('log/train_EMAVFI')
    step = 0
    nr_eval = 0
    best = 0
    dataset = LAVIBDataset('train', data_path, set=tset)
    #sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True)
    args.step_per_epoch = train_data.__len__()
    dataset_val = LAVIBDataset('test', data_path, set=tset)
    val_data = DataLoader(dataset_val, batch_size=batch_size, pin_memory=True, num_workers=8)
    print('training...')
    time_stamp = time.time()
    for epoch in range(60):
        #sampler.set_epoch(epoch)
        bformat='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt}]{postfix}'
    
        with tqdm(total=len(train_data),bar_format=bformat,ascii='░▒█',miniters=1) as pbar:
            for i, imgs in enumerate(train_data):
                data_time_interval = time.time() - time_stamp
                time_stamp = time.time()
                imgs = imgs.to(device, non_blocking=True) / 255.
                imgs, gt = imgs[:, 0:6], imgs[:, 6:]
                learning_rate = get_learning_rate(step)
                _, loss = model.update(imgs, gt, learning_rate, training=True)
                train_time_interval = time.time() - time_stamp
                time_stamp = time.time()
                if step % 200 == 1 and local_rank == 0:
                    writer.add_scalar('learning_rate', learning_rate, step)
                    writer.add_scalar('loss', loss, step)
                if local_rank == 0:
                    pbar.set_postfix_str('epoch:{} time:{:.2f}+{:.2f} loss:{:.4e}'.format(epoch, data_time_interval, train_time_interval, loss))
                    pbar.update()
                    
                step += 1
            nr_eval += 1
        if nr_eval % 3 == 0:
            evaluate(model, val_data, nr_eval, local_rank)
        model.save_model(local_rank)    
            
        #dist.barrier()
@torch.no_grad()
def evaluate(model, val_data, nr_eval, local_rank):
    
    import lpips
    from DISTS_pytorch import DISTS
    from metrics.wattson import WatsonDistanceFft
    from metrics.color_wrapper import ColorWrapper
    from metrics.VSFA import get_vsfa
    
    import pytorch_msssim
    
    
    import metrics.vfips.networks as nets
    
    bformat='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt}]{postfix}'
    
    stats = {'loss':[],'psnr':[],'ssim':[], 'lpips':[], 'dists':[], 'wat_dft':[], 'vsfa':[], 'vfips': []}
    
    loss_fn_squeeze = lpips.LPIPS(net='squeeze').cuda() # LPIPS
    D = DISTS().cuda() # DISTS
    wat_dft = ColorWrapper(WatsonDistanceFft,(),{'reduction':'sum'}).cuda() # Watson-DFT
    vsfa = get_vsfa # VSFA
    
    vfips_net = nets.get_model('multiscale_v33', depth_ksize=1)
    vfips_net.load_state_dict(torch.load('metrics/vfips/exp/model.pytorch')) # VFIPS
    
    
    with tqdm(total=len(val_data),bar_format=bformat,ascii='░▒█',miniters=1) as pbar:
        for i, imgs in enumerate(val_data):
            
            imgs = imgs.to(device, non_blocking=True) / 255.
            imgs, gt = imgs[:, 0:6], imgs[:, 6:]
            pred, _ = model.update(imgs, gt, training=False)
            
            psnr = []
            for j in range(gt.shape[0]):
                psnr.append(-10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item()))
            psnr = np.array(psnr).mean()
            
            ssim = pytorch_msssim.ms_ssim(pred.detach(),gt.detach(), data_range=1, size_average=True).item()
            lpips = loss_fn_squeeze(pred.detach(), gt.detach())
            lpips = lpips.sum().item()/lpips.shape[0]
            dists = D(pred.detach(),gt.detach())
            dists = dists.sum().item()/dists.shape[0]
            w_dft = wat_dft(pred.detach(),gt.detach())/(pred.shape[-2]*pred.shape[-1])
            w_dft = w_dft.item()
            
            vsfa_ = []
            preds = torch.stack([pred for _ in range(12)]).permute(1,0,3,4,2)
            
            for v in preds:
                vsfa_.append(vsfa(v))
            
            #dis = vfips_net(gt.detach().unsqueeze(1).cuda(), pred.detach().unsqueeze(1).cuda())
            #dis = dis.data.detach().cpu().numpy().flatten()
            
            stats['psnr'].append(psnr)
            stats['ssim'].append(ssim)
            stats['lpips'].append(lpips)
            stats['dists'].append(dists)
            stats['wat_dft'].append(w_dft)
            stats['vsfa'].append(sum(vsfa_)/len(vsfa_))
            #stats['vfips'].append(sum(dis)/len(dis))
            
            
            #print(psnr,ssim,lpips,dists,w_dft)
            
            pbar.set_description(f"(val)")
            pbar.set_postfix_str(f" | GPU: {float(gpu_mem_usage()):.2f}G | PSNR: {sum(stats['psnr'])/len(stats['psnr']):.3f} | SSIM: {sum(stats['ssim'])/len(stats['ssim']):.4f} | LPIPS: {sum(stats['lpips'])/len(stats['lpips']):.4f} | DISTS: {sum(stats['dists'])/len(stats['dists']):.4f}  WAT-DFT: {sum(stats['wat_dft'])/len(stats['wat_dft']):.4f} | VSFA: {sum(stats['vsfa'])/len(stats['vsfa']):.4f}  ")
            pbar.update()
            
            
            
    '''
    for _, imgs in enumerate(val_data):
        imgs = imgs.to(device, non_blocking=True) / 255.
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]
        with torch.no_grad():
            pred, _ = model.update(imgs, gt, training=False)
        for j in range(gt.shape[0]):
            psnr.append(-10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item()))
   
    psnr = np.array(psnr).mean()
    if local_rank == 0:
        print(str(nr_eval), psnr)
        writer_val.add_scalar('psnr', psnr, nr_eval)
    '''
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--data_path', type=str, help='data path of lavib', default='/media/SCRATCH/LAVIB')
     
    parser.add_argument('--eval_only' , type=int, default=0,help="Integer for running inference only")
    parser.add_argument('--set', type=str, default='main',choices=['high_afm','low_afm','high_alv','low_alv','high_arl','low_arl','high_arms','low_arms'])
    parser.add_argument("--pretrained" , type=str, help="Load from a pretrained model.")
    
    args = parser.parse_args()
    # torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    if args.local_rank == 0 and not os.path.exists('log'):
        os.mkdir('log')
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank)
    
    if args.pretrained:
        ## For low data, it is better to load from a supervised pretrained model
        loadStateDict = torch.load(args.pretrained)
        modelStateDict = model.net.state_dict()
        for k,v in loadStateDict.items():
            if v.shape == modelStateDict[k].shape:
                print("Loading " , k)
                modelStateDict[k] = v
            else:
                print("Not loading" , k)
        model.net.load_state_dict(modelStateDict)
    if args.eval_only:
        dataset_val = LAVIBDataset('test', args.data_path, set=args.set)
        val_data = DataLoader(dataset_val, batch_size=args.batch_size, pin_memory=True, num_workers=8)
        evaluate(model, val_data, 0, 0)
    else:
        train(model, args.local_rank, args.batch_size, args.data_path, args.set)
        
