import torch
import argparse
import os

from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from utils.data_loading import get_loader, test_dataset
import torch.nn.functional as F
import numpy as np
from seg_metric import cal_mae, cal_fm, cal_sm, cal_em, cal_wfm, cal_dice, cal_iou, cal_ber, cal_acc
from unet.HWTNet import H2SANet


@torch.no_grad()
def test(test_loader, model, epoch, save_path, weight_path, log_f):
    ckpt = torch.load(weight_path, map_location='cuda')

    if hasattr(ckpt, "state_dict"):
        ckpt = ckpt.state_dict()
    model_dict = model.state_dict()

    filtered_ckpt = {k: v for k, v in ckpt.items()
                     if k in model_dict and v.shape == model_dict[k].shape}
    msg = model.load_state_dict(filtered_ckpt, strict=False)

    model.eval()

    mae, fm, sm, em, wfm, m_dice, m_iou, ber, acc = (
        cal_mae(), cal_fm(len(test_loader.dataset)), cal_sm(), cal_em(),
        cal_wfm(), cal_dice(), cal_iou(), cal_ber(), cal_acc()
    )

    for images, gts, names in test_loader:
        images = images.cuda()
        gts = gts.numpy().astype(np.float32)

        res = model(images)
        res = F.interpolate(res, size=gts.shape[-2:], mode='bilinear', align_corners=False)
        res = torch.sigmoid(res).cpu().numpy()

        for i in range(len(res)):
            pred = res[i, 0]
            gt = gts[i, 0]
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            gt /= (gt.max() + 1e-8)
            gt[gt > 0.5] = 1
            gt[gt != 1] = 0
            m_dice.update(pred, gt)
            m_iou.update(pred, gt)

    m_dice_val = m_dice.show()
    m_iou_val = m_iou.show()

    print('M_dice: {:.4f}  M_iou: {:.4f}'.format(m_dice_val, m_iou_val))
    log_f.write(f"  M_dice: {m_dice_val:.4f} | M_iou: {m_iou_val:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=601, help='epoch number, default=30')  
    parser.add_argument('--trainsize', type=int, default=352,  
                        help='the size of training image, try small resolutions for speed (like 256)')
    parser.add_argument('--gpu', type=int, default=0, help='choose which gpu you use')
    parser.add_argument('--save_model', type=str, default='./tn3k_H2SANet/')
    parser.add_argument('--test_img_dir', type=str,
                        default='./data/test/images/')
    parser.add_argument('--test_gt_dir', type=str,
                        default='./data/test/masks/')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    
    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu)

    model = H2SANet(wave_level=2, model_size='mid').cuda()    
    
    best_mae = float('inf')
    best_epoch = 0

    log_path = os.path.join(opt.save_model, f"eval_log_{datetime.now().strftime('%m%d_%H%M')}.txt")
    os.makedirs(opt.save_model, exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")
    log_f.write(f" Test Log ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
    log_f.write(f"Model Path: {opt.save_model}\n")
    log_f.write(f"Epochs: {opt.epoch} | Image Size: {opt.trainsize}\n\n")

    
    epoch_iter = 3
    datasets = ['tn3k']    
    test_root = './data'

    while epoch_iter >= 3 and epoch_iter <= opt.epoch:
        if epoch_iter % 3 == 0 or epoch_iter == opt.epoch:
            print(epoch_iter)
            log_f.write(f"Epoch: {epoch_iter}\n")

            for ds in datasets:
                img_dir = os.path.join(test_root, ds, 'test/images/')
                gt_dir  = os.path.join(test_root, ds, 'test/masks/')
                print(f'--> Testing dataset: {ds}')
                log_f.write(f"Dataset: {ds}\n")

                dataset = test_dataset(img_dir, gt_dir, testsize=opt.trainsize)
                test_loader = DataLoader(dataset, batch_size=25, shuffle=False, num_workers=4)

                weight_path = opt.save_model + 'MedVit_Seg_' + str(epoch_iter) + '.pth'

                test(test_loader, model, epoch_iter, opt.save_model, weight_path, log_f)

        epoch_iter += 3

    log_f.close()
    print(f"\n: {log_path}")
