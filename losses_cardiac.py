import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math


def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :]) 
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :]) 
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1]) 

    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def mse_loss(x, y):
    return torch.mean( (x - y) ** 2 ) 


def diceLoss(y_true, y_pred):
    top = 2 * (y_true * y_pred, [1, 2, 3]).sum()
    bottom = torch.max((y_true + y_pred, [1, 2, 3]).sum(), 50)
    dice = torch.mean(top / bottom)
    return -dice

def diceLoss1(y_true, y_pred):
    top = torch.sum(2 * (y_true * y_pred))
    bottom = torch.sum(y_true)+ torch.sum(y_pred)+1
    dice = torch.mean(top / bottom)
    return dice

# class SoftDiceLoss(y_pred,y_true):
#     __name__ = 'dice_loss'
#
#     def __init__(self, num_classes, activation=None, reduction='mean'):
#         super(SoftDiceLoss, self).__init__()
#         self.activation = activation
#         self.num_classes = num_classes
#
#     def forward(self, y_pred, y_true):
#         class_dice = []
#
#         for i in range(1, self.num_classes):
#             class_dice.append(diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], activation=self.activation))
#         mean_dice = sum(class_dice) / len(class_dice)
#         return 1 - mean_dice

def diceCoeff(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss =  (2 * intersection + eps) / (unionset + eps)

    return loss.sum() / N


def SoftDiceLoss(num_classes,activation,y_pred,y_true):
        class_dice = []

        for i in range(1, num_classes):
            class_dice.append(diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], activation=activation))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice

def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        # equality = np.equal(mask, colour)
        equality=np.array((abs(mask - colour) < 35), dtype=np.bool)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x

def ncc_loss(I, J, win=None):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    conv_fn = getattr(F, 'conv%dd' % ndims)
    I2 = I*I
    J2 = J*J
    IJ = I*J

    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0]/2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1,1)
        padding = (pad_no, pad_no)
    else:
        stride = (1,1,1)
        padding = (pad_no, pad_no, pad_no)
    
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross*cross / (I_var*J_var + 1e-5)

    return -1 * torch.mean(cc)

def lse_loss(I, J, win=None):
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9,9,3]

    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    # pad_no = math.floor(win[0] / 2)

    # if ndims == 1:
    #     stride = (1)
    #     padding = (pad_no)
    # elif ndims == 2:
    #     stride = (1, 1)
    #     padding = (pad_no, pad_no)
    # else:
    # stride = (1, 1, 1)
    padding = (math.floor(win[0] / 2), math.floor(win[1] / 2), math.floor(win[2] / 2))
    stride = (1, 1, 1)
    # padding = 'SAME'

    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, sum_filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, sum_filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, sum_filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, sum_filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, sum_filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    # cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    # I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    # J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc1 = I2_sum + win_size * u_I * u_I - 2 * u_I * I_sum + J2_sum + win_size * u_J * u_J \
          - 2 * u_J * J_sum - 2 * IJ_sum + 2 * u_J * I_sum + 2 * u_I * J_sum - 2 * win_size * u_I * u_J
    return torch.mean(cc1)

def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross, I2_sum,win_size,u_I,