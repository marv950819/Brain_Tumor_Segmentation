import torch
from torch.utils import data
from preprocess import readNpreprocessimage
import numpy as np
import matplotlib.pyplot as plt
import monai
import random
from monai.transforms import Compose, Rotated, RandGaussianNoised, RandFlipd, NormalizeIntensityd, RandShiftIntensityd


class BrainTumorSegDataset(data.Dataset):
    def __init__(self, imgs_pth, lbls_pth, config, mode):
        self.config = config
        self.mode = mode
        self.imgs_pth = imgs_pth
        self.lbls_pth = lbls_pth

        self.transformations= Compose([
            Rotated(keys=["image", "label"], angle=-0.4, mode = ['bilinear', 'nearest']),
            RandGaussianNoised(keys=["image"], prob=0.6),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ])
    
    def __getitem__(self, index):
        img_pth = self.imgs_pth[index]
        lbl_pth = self.lbls_pth[index]
        images = readNpreprocessimage(img_pth, self.config, mask=False)
        mask_lbl = readNpreprocessimage(lbl_pth, self.config, mask=True)
        dict_mask_img = {'image': images, 'label': mask_lbl}
        trans_images = self.transformations(dict_mask_img)

        if lbl_pth['survivaldays'] is not None:
            survival_lbl = int(lbl_pth['survivaldays'])
        else:
            survival_lbl = np.NaN
        return trans_images['image'], trans_images['label'], survival_lbl

    def __len__(self):
        return len(self.imgs_pth)

def visualize(img, mask, filename):
    slice = 75
    fig, ax = plt.subplots(2, 5, figsize=(20, 5))
    img = torch.permute(img, (0,1,3,4,2))
    mask = torch.permute(mask, (0,2,3,1))
    ax[0, 0].imshow(img[0, 0, :, :, slice], cmap='gray'); ax[0, 0].set_title("T2-Flair")
    ax[0, 1].imshow(img[0, 1, :, :, slice], cmap='gray'); ax[0, 1].set_title("T2")
    ax[0, 2].imshow(img[0, 2, :, :, slice], cmap='gray'); ax[0, 2].set_title("T1ce")
    ax[0, 3].imshow(img[0, 3, :, :, slice], cmap='gray'); ax[0, 3].set_title("T1")
    ax[0, 4].imshow(mask[0, :, :, slice], cmap='gray'); ax[0, 4].set_title("Label Mask")

    ax[1, 0].imshow(img[1, 0, :, :, slice], cmap='gray')
    ax[1, 1].imshow(img[1, 1, :, :, slice], cmap='gray')
    ax[1, 2].imshow(img[1, 2, :, :, slice], cmap='gray')
    ax[1, 3].imshow(img[1, 3, :, :, slice], cmap='gray')
    ax[1, 4].imshow(mask[1, :, :, slice], cmap='gray')
    plt.savefig(filename, format='png', bbox_inches='tight')


def get_loader(config, imgs_pth, lbls_pth, mode):
    dataset = BrainTumorSegDataset(imgs_pth, lbls_pth, config, mode)
    if mode == 'train':
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True)
    else:
        data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    # img, mask, lbl = dataset[0]
    # img, mask, lbl = next(iter(data_loader))
    # print(img.shape, mask.shape)
    # visualize(img, mask, "results/Exampleimages.png")
    return data_loader