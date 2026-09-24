import os
import torch
import argparse
from utils.data_loading import get_loader
from datetime import datetime
import torch.nn.functional as F
import logging
import numpy as np
from thop import profile
from unet.HWTNet import H2SANet
import torch.nn as nn


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def train(train_loader, model, optimizer, epoch, opt, loss_func, total_step):
    model.train()
    size_rates = [1] 
    
    for step, data_pack in enumerate(train_loader):
        images, gts = data_pack
        images = images.cuda()
        gts = gts.cuda()
        optimizer.zero_grad()
        x_s_l = model(images)
        if x_s_l.size()[2:] != gts.size()[2:]:
            x_s_l = F.interpolate(x_s_l, size=gts.size()[2:], mode='bilinear', align_corners=True)
        loss_total = loss_func(x_s_l, gts)
        loss_total.backward()
        if opt.clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1) 
        optimizer.step()
        if step % 10 == 0 or step == total_step:
            current_c = F.softplus(model.c_raw).item() + 1e-5
            print('[{}] => [Epoch: {:03d}/{:03d}] => [Step: {:04d}/{:04d}] => Loss: {:.4f} (c: {:.4f})'.
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss_total.data, current_c))
            
            logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                         format(epoch, opt.epoch, step, total_step, loss_total.data))

    if (epoch) % opt.save_epoch == 0:
        save_path = os.path.join(opt.save_model, f'MedVit_Seg_{epoch}.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Checkpoint saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--trainsize', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=30)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_epoch', type=int, default=3)
    parser.add_argument('--save_model', type=str, default='./tn3k_H2SANet/')
    parser.add_argument('--train_img_dir', type=str, default='./data/tn3k/train/images/')
    parser.add_argument('--train_gt_dir', type=str, default='./data/tn3k/train/masks/')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n_skip', type=int, default=3)

    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu)
    os.makedirs(opt.save_model, exist_ok=True)
    
    logging.basicConfig(filename=opt.save_model+'/log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO,
                        filemode='a',
                        datefmt='%Y-%m-%d %I:%M:%S %p')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = H2SANet(wave_level=2, model_size='mid').cuda()
    
    dummy_input = torch.randn(1, 3, opt.trainsize, opt.trainsize).to(device)
    flops, params = profile(model, inputs=(dummy_input, ))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.2f} M")

    head_params_ids = list(map(id, model.seg_head.parameters()))
    curvature_params_ids = [id(model.c_raw)]

    base_params = filter(lambda p: id(p) not in head_params_ids and id(p) not in curvature_params_ids, model.parameters())

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': opt.lr},  
        {'params': model.seg_head.parameters(), 'lr': opt.lr},
        {'params': [model.c_raw], 'lr': opt.lr * 0.1}
    ], weight_decay=1e-4)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch)
    loss_func = structure_loss 
    print(f"Using Loss Function: {loss_func}")

    train_loader = get_loader(opt.train_img_dir, opt.train_gt_dir, batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=8)
    total_step = len(train_loader)
    
    print('-' * 30)
    print("Start Training...")
    
    for epoch_iter in range(1, opt.epoch + 1):
        train(train_loader, model, optimizer, epoch_iter, opt, loss_func, total_step)
        lr_scheduler.step()