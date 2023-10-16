# -*- coding: utf-8 -*-
import os
import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
from datetime import datetime
from models.Swin_Transformer_AnatSwin import DB_SwinTransformer,AnatSwin
from torchvision.utils import make_grid
from data_cardiac import get_loader, val_loader
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options_cardiac_label import opt
import yaml
import losses_cardiac
from losses_cardiac import SoftDiceLoss


cudnn.benchmark = True

Pse_root = opt.Pse_root
gt_root = opt.gt_root
Tem_root = opt.Tem_root

Val_Pse_root = opt.Val_Pse_root
Val_gt_root = opt.Val_gt_root
Val_Tem_root = opt.Val_Tem_root
save_path = opt.save_path

logging.basicConfig(filename=save_path + 'AnatSwin.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("AnatSwin-Train_4_pairs")

model = AnatSwin()

num_parms = 0
if (opt.load is not None):
    model.load_pre(opt.load)
    print('load model from ', opt.load)

model.cuda()
for p in model.parameters():
    num_parms += p.numel()
logging.info("Total Parameters (For Reference): {}".format(num_parms))
print("Total Parameters (For Reference): {}".format(num_parms))

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loader(Pse_root, gt_root,Tem_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
valdata_loader = val_loader(Val_Pse_root, Val_gt_root,Val_Tem_root, opt.trainsize)
total_step = len(train_loader)

logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0
best_mae1 = 1
best_epoch1 = 0
best_mae2 = 1
best_epoch2 = 0
sim_loss_fn = losses_cardiac.mse_loss

# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()

    loss_all = 0
    epoch_step = 0

    try:
        for i, (pseimgs, gts, temimgs) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            pseimgs = pseimgs.cuda()
            gts = gts.cuda()
            temimgs = temimgs.repeat(1,3,1,1).cuda()
            s= model(pseimgs,temimgs)

            num_classes = 4
            activation = 'sigmoid'
            sal_loss= SoftDiceLoss(num_classes,activation, s, gts)
            loss = sal_loss
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}||sal_loss:{:4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             optimizer.state_dict()['param_groups'][0]['lr'], sal_loss.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f} , mem_use:{:.0f}MB'.
                        format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'], sal_loss.data,memory_used))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(pseimgs[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'AnatSwin_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'AnatSwin_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise

def bce2d_new(input, target, reduction=None):
    assert (input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

# test function
def val(val_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0

        def mask_class(seg, value):
            a = np.array((abs(seg - value) < 0.5), dtype=np.float64)
            a=torch.tensor(a)
            return a

        def mask_class4(seg, value):
            a = np.array((abs(seg - value) < 0.14 ), dtype=np.float64)
            a = torch.tensor(a)
            return a

        for i in range(val_loader.size):
            pseimg, gt, temimg, name, img_for_post = val_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            pseimg = pseimg.cuda()
            temimg = temimg.repeat(1, 3, 1, 1).cuda()
            res = model(pseimg, temimg)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            mae_lv = losses_cardiac.diceLoss1(mask_class(res[3, :, :], 1), mask_class4(gt, 1)).numpy()
            mae_rv = losses_cardiac.diceLoss1(mask_class(res[1, :, :], 1), mask_class4(gt, 0.27)).numpy()
            mae_my = losses_cardiac.diceLoss1(mask_class(res[2, :, :], 1), mask_class4(gt, 0.54)).numpy()
            ave_dice = (mae_lv + mae_rv + mae_my) / 3.0
            mae_sum += ave_dice

        mae = 1-mae_sum / val_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'AnatSwin_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        val(valdata_loader, model, epoch, save_path)
