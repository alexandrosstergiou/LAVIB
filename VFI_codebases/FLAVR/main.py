import os, psutil
import sys
import time
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import random
import config
import myutils
from loss import Loss
from torch.utils.data import DataLoader

import pytorch_msssim

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

def load_checkpoint(args, model, optimizer , path):
    print("loading checkpoint %s" % path)
    checkpoint = torch.load(path)
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = checkpoint.get("lr" , args.lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

##### Parse CmdLine Arguments #####
args, unparsed = config.get_args()
cwd = os.getcwd()
print(args)

save_loc = os.path.join(args.checkpoint_dir , "saved_models_final" , args.dataset , args.exp_name, args.set)
if not os.path.exists(save_loc):
    os.makedirs(save_loc)
opts_file = os.path.join(save_loc , "opts.txt")
with open(opts_file , "w") as fh:
    fh.write(str(args))


##### TensorBoard & Misc Setup #####
writer_loc = os.path.join(args.checkpoint_dir , 'tensorboard_logs_%s_final/%s' % (args.dataset , args.exp_name))
writer = SummaryWriter(writer_loc)

device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


if args.dataset == "vimeo90K_septuplet":
    from dataset.vimeo90k_septuplet import get_loader
    train_loader = get_loader('train', args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers) 
elif args.dataset == "lavib":
    from dataset.lavib import get_loader
    train_loader = get_loader('train', args.data_root, args.batch_size, shuffle=False, num_workers=args.num_workers, set=args.set)
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers, set=args.set)   
elif args.dataset == "gopro":
    from dataset.GoPro import get_loader
    train_loader = get_loader(args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers, test_mode=False, interFrames=args.n_outputs, n_inputs=args.nbr_frame)
    test_loader = get_loader(args.data_root, args.batch_size, shuffle=False, num_workers=args.num_workers, test_mode=True, interFrames=args.n_outputs, n_inputs=args.nbr_frame)
else:
    raise NotImplementedError


from model.FLAVR_arch import UNet_3D_3D
print("Building model: %s"%args.model.lower())
model = UNet_3D_3D(args.model.lower() , n_inputs=args.nbr_frame, n_outputs=args.n_outputs, joinType=args.joinType, upmode=args.upmode)
model = torch.nn.DataParallel(model).to(device)


##### Define Loss & Optimizer #####
criterion = Loss(args)

## ToDo: Different learning rate schemes for different parameters
from torch.optim import Adam
optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

#@torch.autocast(device_type="cuda")
def train(args, epoch):
    model.train()
    criterion.train()

    stats = {'loss':[],'psnr':[],'ssim':[]}
    
    bformat='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt}]{postfix}'
    
    with tqdm(total=len(train_loader),bar_format=bformat,ascii='░▒█',miniters=1) as pbar:
        for i, (images, gt)  in enumerate(train_loader):
            

            # Build input batch
            images = torch.stack(images , dim=2).cuda()/255.
            gt = gt.cuda()/255.
            # Forward
            optimizer.zero_grad()
            out = model(images)[0]
            

            loss, loss_specific = criterion(out, gt)
            
            # Save loss values
            stats['loss'].append(loss.detach().item())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()
            
            psnr = calc_psnr(out.detach(),gt.detach())
            
            if psnr < 0:
                print(loss.item())
                sys.exit()
            
            ssim = pytorch_msssim.ms_ssim(out.detach(),gt.detach(), data_range=1, size_average=True).item()
            
            stats['psnr'].append(psnr)
            stats['ssim'].append(ssim)
            
            
            pbar.set_description(f"(train) EPOCH: {epoch:02d}")
            ram = cpu_mem_usage()
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix_str(f" | GPU: {float(gpu_mem_usage()):.2f}G | RAM: {ram[0]:.2f}G | LR : {lr:.1e} | Loss: {loss.detach():.3e}/{sum(stats['loss'])/len(stats['loss']):.3e} | PSNR: {psnr:.3f}/{sum(stats['psnr'])/len(stats['psnr']):.3f} | SSIM: {ssim:.3f}/{sum(stats['ssim'])/len(stats['ssim']):.4f}")
            pbar.update()
    
    return sum(stats['loss'])/len(stats['loss']), sum(stats['psnr'])/len(stats['psnr'])
            


@torch.no_grad()
def test(args, epoch):
    import lpips
    print('Evaluating for epoch = %d' % epoch)
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()
    criterion.eval()
    
    stats = {'loss':[],'psnr':[],'ssim':[], 'lpips':[]}
    
    bformat='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt}]{postfix}'
    
    loss_fn_squeeze = lpips.LPIPS(net='squeeze').cuda()
    
    with tqdm(total=len(test_loader),bar_format=bformat,ascii='░▒█') as pbar:
        for i, (images, gt) in enumerate(test_loader):

            # Build input batch
            images = torch.stack(images , dim=2).cuda()/255.
            gt = gt.cuda()/255.

            out = model(images)[0] ## images is a list of neighboring frames
            loss, _ = criterion(out, gt)
            
            
            # Save loss values
            stats['loss'].append(loss.detach().item())
            
            psnr = calc_psnr(out.detach(),gt.detach())
            ssim = pytorch_msssim.ms_ssim(out.detach(),gt.detach(), data_range=1, size_average=True).item()
            lpips = loss_fn_squeeze(out.detach(), gt.detach())
            lpips = lpips.sum().item()/lpips.shape[0]
            
            stats['psnr'].append(psnr)
            stats['ssim'].append(ssim)
            stats['lpips'].append(lpips)
            
            
            pbar.set_description(f"(val) EPOCH: {epoch:02d}")
            ram = cpu_mem_usage()
            pbar.set_postfix_str(f" | GPU: {float(gpu_mem_usage()):.2f}G | RAM: {ram[0]:.2f}G | Loss: {loss.detach():.3e}/{sum(stats['loss'])/len(stats['loss']):.3e} | PSNR: {psnr:.3f}/{sum(stats['psnr'])/len(stats['psnr']):.3f} | SSIM: {ssim:.3f}/{sum(stats['ssim'])/len(stats['ssim']):.4f} | LPIPS: {lpips:.3f}/{sum(stats['lpips'])/len(stats['lpips']):.4f}")
            pbar.update()

        
    loss_avg = sum(stats['loss'])/len(stats['loss'])
    psnr_avg = sum(stats['psnr'])/len(stats['psnr'])              
    ssim_avg = sum(stats['ssim'])/len(stats['ssim'])
    lpips_avg = sum(stats['lpips'])/len(stats['lpips'])
    
    
    print('--------')
    print(f"Loss: {loss_avg}")
    print(f"PSNR: {psnr_avg}")
    print(f"SSIM: {ssim_avg}")
    print(f"LPIPS: {lpips_avg}")
    
    

    # Save psnr & ssim
    '''
    save_fn = os.path.join(save_loc, 'results.txt')
    with open(save_fn, 'a') as f:
        f.write('For epoch=%d\t' % epoch)
        f.write("PSNR: %f, SSIM: %f\n" %
                (psnr_avg, ssim_avg))

    # Log to TensorBoard
    timestep = epoch +1
    writer.add_scalar('Loss/test', loss_avg, timestep)
    writer.add_scalar('PSNR/test', psnr_avg, timestep)
    writer.add_scalar('SSIM/test', ssim_avg, timestep)
    '''
    return loss_avg, psnr_avg, ssim_avg


""" Entry Point """
def main(args):

    if args.pretrained:
        ## For low data, it is better to load from a supervised pretrained model
        loadStateDict = torch.load(args.pretrained)
        modelStateDict = model.state_dict()

        for k,v in loadStateDict.items():
            if v.shape == modelStateDict[k].shape:
                print("Loading " , k)
                modelStateDict[k] = v
            else:
                print("Not loading" , k)

        model.load_state_dict(modelStateDict)

    best_psnr = 0
    
    if args.eval_only:
        args.start_epoch = 0
        args.max_epoch = 1
    
    for epoch in range(args.start_epoch, args.max_epoch):
        
        if not args.eval_only:
            loss, psnr = train(args, epoch)
            torch.save(model.state_dict(), save_loc+f'/model_{epoch}.pth')
        else:
            test_loss, psnr, _ = test(args, epoch)
            loss = test_loss

        # save checkpoint
        #is_best = psnr > best_psnr
        #best_psnr = max(psnr, best_psnr)
        #myutils.save_checkpoint({
        #    'epoch': epoch,
        #    'state_dict': model.state_dict(),
        #    'optimizer': optimizer.state_dict(),
        #    'best_psnr': best_psnr,
        #    'lr' : optimizer.param_groups[-1]['lr']
        #}, save_loc, is_best, args.exp_name)

        # update optimizer policy
        scheduler.step(loss)

if __name__ == "__main__":
    main(args)
