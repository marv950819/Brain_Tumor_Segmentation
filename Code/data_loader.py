from torch.utils import data
from monai.transforms import LoadImaged, Compose, ScaleIntensityRanged, SpatialCropd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import torch
import monai
from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, CropForeground, ToTensord, RandAffined

class BrainTumorSegDataset(data.Dataset):
    def __init__(self, imgs_pth, lbls_pth, config, mode):
        self.config = config
        self.mode = mode
        self.imgs_pth = imgs_pth
        self.lbls_pth = lbls_pth

        self.img_transforms = Compose([
            LoadImaged(keys=['t2flair', 't1', 't1ce', 't2']),
            ScaleIntensityRanged(keys=['t2flair', 't1', 't1ce', 't2'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ToTensord(keys=['t2flair', 't1', 't1ce', 't2']),
            # Note: CropForeground is not working properly, should accept 3 parameters on roi_size neither CropForeground or RandAffined
            #CenterSpatialCrop(keys=['mask'], roi_size=(config.slice_depth_size, config.image_size, config.image_size)),
            #CropForeground(keys=['t2flair', 't1', 't1ce', 't2'],  roi_size=(config.slice_depth_size, config.image_size, config.image_size), allow_smaller=True),
            # RandAffined(
            #     keys=['t2flair', 't1', 't1ce', 't2'],
            #     prob=1.0,
            #     spatial_size=(300, 300, 50),
            #     translate_range=(40, 40, 2),
            #     rotate_range=(np.pi / 36, np.pi / 36, np.pi / 4),
            #     scale_range=(0.15, 0.15, 0.15),
            #     padding_mode="border",
            # ),
            ])
        
        self.mask_transforms = Compose([
            LoadImaged(keys=['mask']),
            ScaleIntensityRanged(keys=['mask'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ToTensord(keys=['mask']),
            #CenterSpatialCrop(keys=['mask'], roi_size=(128, 128, 128)),
        ])

    def __getitem__(self, index):

        # Load and transform images
        img_data = self.img_transforms({key: self.imgs_pth[index][key] for key in ['t2flair', 't1', 't1ce', 't2']})
        
        # stack images
        images = torch.stack([
            img_data[key]
            for key in ['t2flair', 't1', 't1ce', 't2']], dim=0)
        
        # Load and transfrom mask
        mask = self.mask_transforms({key: self.lbls_pth[index][key] for key in ['mask']})['mask']
    
        # survival label
        survival_lbl = np.nan
        if 'survivaldays' in self.lbls_pth[index] and self.lbls_pth[index]['survivaldays'] is not None:
            survival_lbl = int(self.lbls_pth[index]['survivaldays'])

        return images, mask, survival_lbl

    def __len__(self):
        return len(self.imgs_pth)


def visualize(img, mask):
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
    plt.savefig("Exampleimages_4.png", format='png', bbox_inches='tight')

def get_loader(config, imgs_pth, lbls_pth, mode):
    dataset = BrainTumorSegDataset(imgs_pth, lbls_pth, config, mode)
    if mode == 'train':
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True)
    else:
        data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    # img, mask, lbl = dataset[0]
    # img, mask, lbl = next(iter(data_loader))
    # visualize(img, mask)
    # print(img.shape, mask.shape, lbl.shape)

    return data_loader

