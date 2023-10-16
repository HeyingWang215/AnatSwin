# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=500, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default='./swin_base_patch4_window12_384_22k.pth', help='train from checkpoints')
parser.add_argument('--load_pre', type=str, default='./cardiac_cpts/AnatSwin_epoch_best.pth', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--Pse_root', type=str, default='../', help='the training Pse images root')
parser.add_argument('--Tem_root', type=str, default='../', help='the training Tem images root')
parser.add_argument('--gt_root', type=str, default='../', help='the training gt images root')
parser.add_argument('--Val_Pse_root', type=str, default='../', help='the val Pse images root')
parser.add_argument('--Val_Tem_root', type=str, default='../', help='the val Tem images root')
parser.add_argument('--Val_gt_root', type=str, default='../', help='the val gt images root')
parser.add_argument('--save_path', type=str, default='./cardiac_cpts/', help='the path to save models and logs')
opt = parser.parse_args()