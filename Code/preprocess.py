import SimpleITK as sitk
import numpy as np
import torchvision.transforms as transforms
import torch


def readNpreprocessimage(imgs_pth, config, mask=False):
    if not mask:
        finimage = {}
        for keys in imgs_pth.keys():
            temp = sitk.Cast(sitk.RescaleIntensity(sitk.ReadImage(imgs_pth[keys]), 0, 255), sitk.sitkUInt8)
            # Can fit in different preprocessing if required
            temp = np.moveaxis(sitk.GetArrayFromImage(temp), 0, -1)
            temp = transforms.ToTensor()(temp)
            temp = temp[config.start_pos[0]:config.start_pos[0]+config.slice_depth_size,
                        config.start_pos[1]:config.start_pos[1]+config.image_size,
                        config.start_pos[2]:config.start_pos[2]+config.image_size]
            # finimage[keys] = torch.permute(temp, (1,2,0))
            finimage[keys] = temp
        # fusedimage size (channels, depth, height, width)
        fusedimage = torch.stack((finimage['t2flair'], finimage['t2'], finimage['t1ce'], finimage['t1']), dim=0)
    else:
        # maskimage = np.moveaxis(sitk.GetArrayFromImage(sitk.ReadImage(imgs_pth['mask'])), 0, -1)
        maskimage = sitk.GetArrayFromImage(sitk.ReadImage(imgs_pth['mask']))

        # Whole Tumor
        maskimage[maskimage == 2] = 1
        maskimage[maskimage == 4] = 1

        # Tumor Core
        # maskimage[maskimage == 2] = 0
        # maskimage[maskimage == 4] = 1

        # Enhancing Tumor
        # maskimage[maskimage == 1] = 0
        # maskimage[maskimage == 2] = 0
        # maskimage[maskimage == 4] = 1

        maskimage = maskimage.astype(np.uint8)
        maskimage = maskimage[config.start_pos[0]:config.start_pos[0]+config.slice_depth_size,
                              config.start_pos[1]:config.start_pos[1]+config.image_size,
                              config.start_pos[2]:config.start_pos[2]+config.image_size]
        # maskimage = np.moveaxis(maskimage, 0, -1)
        # fusedimage size (depth height, width)
        fusedimage = torch.from_numpy(maskimage).long()
    return fusedimage


#Preprocessing Steps
# 1) Histogram Equalization: https://simpleitk.readthedocs.io/en/master/link_SliceBySliceDecorator_docs.html
# 2) Bias Field Correction: https://medium.com/@alexandro.ramr777/how-to-do-bias-field-correction-with-python-156b9d51dd79
# 3) Denoising Filter: https://simpleitk.readthedocs.io/en/master/filters.html
# 4) Intensity normalization: https://medium.com/@sumitgulati59/brain-tumor-segmentation-b97de6619e04 (Nice Link)