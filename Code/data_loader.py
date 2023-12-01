import torch
from torch.utils import data
import numpy as np
import SimpleITK as sitk
import monai
from monai.transforms import LoadImaged, Compose, ScaleIntensityRanged, SpatialCropd
import matplotlib.pyplot as plt

class BrainTumorSegDataset(monai.data.Dataset):
    def __init__(self, imgs_pth, lbls_pth, config, mode):
        self.config = config
        self.mode = mode
        self.imgs_pth = imgs_pth
        self.lbls_pth = lbls_pth

        #end positions
        end_pos_0 = config.start_pos[0] + config.slice_depth_size
        end_pos_1 = config.start_pos[1] + config.image_size
        end_pos_2 = config.start_pos[2] + config.image_size

        self.transforms = Compose([
            LoadImaged(keys=['t2flair', 't1', 't1ce', 't2']),
            ScaleIntensityRanged(keys=['t2flair', 't1', 't1ce', 't2'], a_min=0, a_max=255),
            SpatialCropd(keys=['t2flair', 't1', 't1ce', 't2'], roi_start=[config.start_pos[0],config.start_pos[1],config.start_pos[2]], roi_end=[end_pos_0, end_pos_1, end_pos_2]),
        ])


    def __getitem__(self, index):
        img_data = self.transforms({key: self.imgs_pth[index][key] for key in ['t2flair', 't1', 't1ce', 't2']})
        print("Image shape after transformation:", img_data['t1'].shape)
        images = torch.stack([img_data['t2flair'], img_data['t1'], img_data['t1ce'], img_data['t2']], dim=0)

        mask_lbl = self.process_mask(self.lbls_pth[index]['mask']) if 'mask' in self.lbls_pth[index] else None

        survival_lbl = np.nan
        if 'survivaldays' in self.lbls_pth[index] and self.lbls_pth[index]['survivaldays'] is not None:
            survival_lbl = int(self.lbls_pth[index]['survivaldays'])

        return images, mask_lbl, survival_lbl

    def process_mask(self, mask_path):
        maskimage = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))

        # Whole Tumor
        maskimage[maskimage == 2] = 1
        maskimage[maskimage == 4] = 1

        # Apply additional mask processing if required

        maskimage = maskimage.astype(np.uint8)
        # maskimage = maskimage[self.config.start_pos[0]:self.config.start_pos[0]+self.config.slice_depth_size,
        #                       self.config.start_pos[1]:self.config.start_pos[1]+self.config.image_size,
        #                       self.config.start_pos[2]:self.config.start_pos[2]+self.config.image_size]
        print("Mask shape after cropping:", maskimage.shape)
        return torch.from_numpy(maskimage).long()
    

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
    plt.savefig("Exampleimages_1.png", format='png', bbox_inches='tight')

def get_loader(config, imgs_pth, lbls_pth, mode):
    dataset = BrainTumorSegDataset(imgs_pth, lbls_pth, config, mode)
    if mode == 'train':
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True)
    else:
        data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    img, mask, lbl = dataset[0]
    img, mask, lbl = next(iter(data_loader))
    visualize(img, mask)
    print(img.shape, mask.shape, lbl.shape)
    print(error)
    return data_loader

