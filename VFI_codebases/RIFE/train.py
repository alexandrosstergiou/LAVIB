import os, sys, psutil
import cv2
import math
import time
import torch
import numpy as np
import random
import argparse
from tqdm import tqdm

from model.RIFE import Model
from dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import pytorch_msssim

device = torch.device("cuda")
log_path = 'train_log'
if not os.path.isdir(log_path):
    os.makedirs(log_path)

def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024 ** 3

def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total

def calc_psnr(pred, gt, mask=None):
    '''
        Here we assume quantized(0-1.) arguments.
    '''
    diff = (pred - gt)

    if mask is not None:
        mse = diff.pow(2).sum() / (3 * mask.sum())
    else:
        mse = diff.pow(2).mean() + 1e-8    # mse can (surprisingly!) reach 0, which results in math domain error

    return -10 * math.log10(mse)

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

#@torch.autocast(device_type="cuda")
def train(model, local_rank, root_dir='/media/SCRATCH/LAVIB', s='main'):
    if local_rank == 0:
        writer = SummaryWriter('train')
        writer_val = SummaryWriter('validate')
    else:
        writer = None
        writer_val = None
    step = 0
    nr_eval = 0
    dataset = LAVIBDataset('train',data_root=root_dir,set=s)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True)
    args.step_per_epoch = train_data.__len__()
    dataset_val = LAVIBDataset('test',data_root=root_dir,set=s)
    val_data = DataLoader(dataset_val, batch_size=8, pin_memory=True, num_workers=8)
    print('training...')
    time_stamp = time.time()
    
    bformat='{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}'
    
    stats = {'loss':[],'psnr':[],'ssim':[]}
    
    for epoch in range(args.epoch):
         with tqdm(total=len(train_data),bar_format=bformat,ascii='░▒█') as pbar:
            for i, data in enumerate(train_data):
                
                data_time_interval = time.time() - time_stamp
                time_stamp = time.time()
                data_gpu, timestep = data
                data_gpu = data_gpu.to(device, non_blocking=True) / 255.
                timestep = timestep.to(device, non_blocking=True)
                imgs = data_gpu[:, :6]
                gt = data_gpu[:, 6:9]
                learning_rate = get_learning_rate(step) / 4
                pred, info = model.update(imgs, gt, learning_rate, training=True) # pass timestep if you are training RIFEm
                
                psnr = calc_psnr(gt,pred) 
                ssim = pytorch_msssim.ms_ssim(pred.detach(), gt.detach(), data_range=1, size_average=True).item()
                
                stats['psnr'].append(psnr)
                stats['ssim'].append(ssim)
                
                time_stamp = time.time()
                if step % 10000 == 1 and local_rank != 0:
                    writer.add_scalar('learning_rate', learning_rate, step)
                    writer.add_scalar('loss/l1', info['loss_l1'].detach(), step)
                    writer.add_scalar('loss/tea', info['loss_tea'].detach(), step)
                    writer.add_scalar('loss/distill', info['loss_distill'].detach(), step)
                if step % 20000 == 1 and local_rank != 0:
                    gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                    mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                    pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                    merged_img = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                    flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                    flow1 = info['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()
                    
                step += 1
                
                pbar.set_description(f"EPOCH: {epoch:02d}")
                stats['loss'].append(info['loss_l1'])
                ram = cpu_mem_usage()
                pbar.set_postfix_str(f" | GPU: {float(gpu_mem_usage()):.2f}G | RAM: {ram[0]:.2f}G | Loss: {info['loss_l1']:.3f}/{sum(stats['loss'])/len(stats['loss']):.3f} | PSNR: {psnr:.3f}/{sum(stats['psnr'])/len(stats['psnr']):.3f} | SSIM: {ssim:.3f}/{sum(stats['ssim'])/len(stats['ssim']):.4f}")
                pbar.update()
                
            nr_eval += 1
            if nr_eval % 1 == 0:
                evaluate(model, val_data, step, local_rank, writer_val)
            model.save_model(log_path, local_rank)    

@torch.no_grad()
def evaluate(model, val_data, nr_eval, local_rank, writer_val):
    import lpips
    from DISTS_pytorch import DISTS    
    from metrics.wattson import WatsonDistanceFft
    from metrics.color_wrapper import ColorWrapper
    from metrics.VSFA import get_vsfa
    
    import metrics.vfips.networks as nets
    
    loss_l1_list = []
    loss_distill_list = []
    loss_tea_list = []
    psnr_list = []
    psnr_list_teacher = []
    
    stats = {'loss':[],'psnr':[],'ssim':[], 'lpips':[], 'dists':[], 'wat_dft':[], 'vsfa':[], 'vfips': []}

    bformat='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt}]{postfix}'
    
    loss_fn_squeeze = lpips.LPIPS(net='squeeze').cuda() # LPIPS
    D = DISTS().cuda() # DISTS
    wat_dft = ColorWrapper(WatsonDistanceFft,(),{'reduction':'sum'}).cuda() # Watson-DFT
    vsfa = get_vsfa # VSFA
    
    #vfips_net = nets.get_model('multiscale_v33', depth_ksize=1)
    #vfips_net.load_state_dict(torch.load('metrics/vfips/exp/model.pytorch')) # VFIPS
    
    with tqdm(total=len(val_data),bar_format=bformat,ascii='░▒█') as pbar:
        for i, data in enumerate(val_data):
            
            time_stamp = time.time()
            data_gpu, timestep = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            timestep = timestep.to(device, non_blocking=True)
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            pred, info = model.update(imgs, gt, training=False) 
            
            psnr = calc_psnr(gt.detach(),pred.detach()) 
            ssim = pytorch_msssim.ms_ssim(pred.detach(), gt.detach(), data_range=1, size_average=True).item()
            
            lpips = loss_fn_squeeze(pred.detach(), gt.detach())
            lpips = lpips.sum().item()/lpips.shape[0]
            dists = D(pred.detach(),gt.detach())
            dists = dists.sum().item()/dists.shape[0]
            w_dft = wat_dft(pred.detach(),gt.detach())/(pred.shape[-2]*pred.shape[-1])
                    
            stats['psnr'].append(psnr)
            stats['ssim'].append(ssim)
            stats['lpips'].append(lpips)
            stats['dists'].append(dists)
            stats['wat_dft'].append(w_dft)
            
            pbar.set_description(f"(val)")
            ram = cpu_mem_usage()
            pbar.set_postfix_str(f" | GPU: {float(gpu_mem_usage()):.2f}G | RAM: {ram[0]:.2f}G | PSNR: {sum(stats['psnr'])/len(stats['psnr']):.3f} | SSIM: {sum(stats['ssim'])/len(stats['ssim']):.4f} | LPIPS: {sum(stats['lpips'])/len(stats['lpips']):.4f} | DISTS: {sum(stats['dists'])/len(stats['dists']):.4f} |  WAT-DFT: {sum(stats['wat_dft'])/len(stats['wat_dft']):.4f}")
            pbar.update()
            
            #merged_img = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
            
    
    eval_time_interval = time.time() - time_stamp
    print(f"(Eval) PSNR: {sum(stats['psnr'])/len(stats['psnr']):.4f} SSIM: {sum(stats['ssim'])/len(stats['ssim']):.4f}")
    if local_rank != 0:
        return
    #writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    #writer_val.add_scalar('psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--root_dir', type=str, help='data dir')
     
    parser.add_argument('--eval_only' , type=int, default=0,help="Integer for running inference only")
    parser.add_argument('--set', type=str, default='main',choices=['high_afm','low_afm','high_alv','low_alv','high_arl','low_arl','high_arms','low_arms'])
    parser.add_argument("--pretrained" , type=str, help="Load from a pretrained model.")
    
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
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
        modelStateDict = model.flownet.state_dict()
        for k,v in loadStateDict.items():
            if v.shape == modelStateDict[k].shape:
                print("Loading " , k)
                modelStateDict[k] = v
            else:
                print("Not loading" , k)
        model.flownet.load_state_dict(modelStateDict)
    
    if args.eval_only:
        dataset_val = LAVIBDataset('test',data_root=args.root_dir,set=args.set)
        val_data = DataLoader(dataset_val, batch_size=2, pin_memory=True, num_workers=8)
        evaluate(model,val_data,1,args.local_rank,None)
    else: 
        train(model, args.local_rank,root_dir=args.root_dir,s=args.set)
        
