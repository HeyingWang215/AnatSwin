import os
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import natsort
import numpy as np
from PIL import ImageEnhance
import losses_cardiac
from losses_cardiac import mask_to_onehot

# several data augumentation strategies
def cv_random_flip(Pse, label, Tem):
    flip_flag = random.randint(0, 1)

    if flip_flag == 1:
        Pse = Pse.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        Tem = Tem.transpose(Image.FLIP_LEFT_RIGHT)

    return Pse, label, Tem

def randomCrop(Pse, label, Tem):
    border = 30
    image_width = Pse.size[0]
    image_height = Pse.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return Pse.crop(random_region), label.crop(random_region), Tem.crop(random_region)

def randomRotation1(Pse, r,g,b,a, Tem):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        Pse = Pse.rotate(random_angle, mode)
        r = r.rotate(random_angle, mode)
        g = g.rotate(random_angle, mode)
        b = b.rotate(random_angle, mode)
        a = a.rotate(random_angle, mode)
        Tem = Tem.rotate(random_angle, mode)
    return Pse, r,g,b,a, Tem

def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)

class trainDataset(data.Dataset):
    def __init__(self, pse_root, gt_root, tem_root, trainsize):
        self.trainsize = trainsize
        self.pseimgs = [pse_root + f for f in os.listdir(pse_root) if f.endswith('.jpg')
                       or f.endswith('.png')]

        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.temimgs = [tem_root + f for f in os.listdir(tem_root) if f.endswith('.bmp')
                       or f.endswith('.png') or f.endswith('.jpg')]

        self.temimgs = natsort.natsorted(self.temimgs)
        self.gts = natsort.natsorted(self.gts)
        self.pseimgs = natsort.natsorted(self.pseimgs)
        self.size = len(self.pseimgs)
        self.pseimgs_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [1, 1, 1])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.temimgs_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)),
             transforms.ToTensor(),
             transforms.Normalize([0], [1])])

    def __getitem__(self, index):
        pse = self.rgb_loader(self.pseimgs[index])
        gt = self.binary_loader(self.gts[index])
        indexim=int(index / 900 )* 12
        if index<900:
            tem = self.binary_loader(self.temimgs[index % 12])
        else:
            tem = self.binary_loader(self.temimgs[index % 12+indexim])

        gt1=np.array(gt)
        gt1 = np.expand_dims(gt1, axis=2)
        palette= [[0], [70], [140], [255]]
        gtim=mask_to_onehot(gt1, palette)
        gtim=gtim*255
        gtim = Image.fromarray(np.uint8(gtim))
        gt=gtim
        pse, gt, tem = cv_random_flip(pse, gt, tem)
        pse, gt, tem = randomCrop(pse, gt, tem)
        r, g, b,a = gt.split()
        pse, r,g,b,a,tem = randomRotation1(pse, r,g,b,a, tem)
        gt = Image.merge("RGBA", [r,g,b,a])

        gt = randomPeper(gt)

        pse = self.pseimgs_transform(pse)
        r, g, b, a = gt.split()
        r1 = self.gt_transform(r)
        g1 = self.gt_transform(g)
        b1 = self.gt_transform(b)
        a1 = self.gt_transform(a)
        gt = torch.cat((r1,g1,b1,a1), 0)
        tem = self.temimgs_transform(tem)

        return pse, gt, tem

    def filter_files(self):
        assert len(self.pseimgs) == len(self.gts) and len(self.gts) == len(self.pseimgs)
        pseimgs = []
        gts = []
        temimgs = []
        for img_path, gt_path, tem_path in zip(self.pseimgs, self.gts, self.temimgs):
            pse = Image.open(pse_path)
            gt = Image.open(gt_path)
            tem = Image.open(tem_path)
            if pse.size == gt.size and gt.size == tem.size :
                pseimgs.append(pse_path)
                gts.append(gt_path)
                temimgs.append(tem_path)
        self.pseimgs = pseimgs
        self.gts = gts
        self.temimgs = temimgs

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, pse, gt, tem):
        assert pse.size == gt.size and gt.size == tem.size
        w, h = pse.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return pse.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), \
                   tem.resize((w, h),Image.NEAREST)
        else:
            return pse, gt, tem

    def __len__(self):
        return self.size

# dataloader for training
def get_loader(Pse_root, gt_root, Tem_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):
    dataset = trainDataset(Pse_root, gt_root, Tem_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

# val dataset and loader
class val_loader:
    def __init__(self, pse_root, gt_root, tem_root, testsize):
        self.testsize = testsize
        self.pseimgs = [pse_root + f for f in os.listdir(pse_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.temimgs = [tem_root + f for f in os.listdir(tem_root) if f.endswith('.bmp')
                       or f.endswith('.png')or f.endswith('.jpg')]

        self.pseimgs = natsort.natsorted(self.pseimgs)
        self.gts = natsort.natsorted(self.gts)
        self.temimgs = natsort.natsorted(self.temimgs)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [1, 1, 1])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.pse_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor(),transforms.Normalize([0],[1])])
        self.size = len(self.pseimgs)
        self.index = 0

    def load_data(self):
        pse = self.rgb_loader(self.pseimgs[self.index])
        pse = self.transform(pse).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])

        if self.index<900:
            tem = self.binary_loader(self.temimgs[self.index % 12])
        else:
            tem = self.binary_loader(self.temimgs[self.index % 12+12])
        tem = self.pse_transform(tem).unsqueeze(0)
        name = self.pseimgs[self.index].split('/')[-1]
        image_for_post = self.rgb_loader(self.pseimgs[self.index])
        image_for_post = image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size

        return pse, gt, tem, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


