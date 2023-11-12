import SimpleITK as sitk
import numpy as np
import torchvision.transforms as transforms
import torch


def readNpreprocessimage(imgs_pth, mask=False):
    if not mask:
        finimage = {}
        for keys in imgs_pth.keys():
            temp = sitk.Cast(sitk.RescaleIntensity(sitk.ReadImage(imgs_pth[keys]), 0, 255), sitk.sitkUInt8)
            # Can fit in different preprocessing if required
            temp = np.moveaxis(sitk.GetArrayFromImage(temp), 0, -1)
            temp = transforms.ToTensor()(temp)
            temp = temp[45:109, 88:152, 88:152]
            finimage[keys] = torch.permute(temp, (1,2,0))
        # fusedimage size (channels, height, width, depth)
        fusedimage = torch.stack((finimage['t2flair'], finimage['t2'], finimage['t1ce'], finimage['t1']), dim=0)
    else:
        # maskimage = np.moveaxis(sitk.GetArrayFromImage(sitk.ReadImage(imgs_pth['mask'])), 0, -1)
        maskimage = sitk.GetArrayFromImage(sitk.ReadImage(imgs_pth['mask']))
        maskimage[maskimage == 4] = 3
        maskimage = maskimage.astype(np.uint8)
        maskimage = maskimage[45:109, 88:152, 88:152]
        maskimage = np.moveaxis(maskimage, 0, -1)
        # fusedimage size (height, width, depth)
        fusedimage = torch.from_numpy(maskimage).long()
    return fusedimage


#Preprocessing Steps
# 1) Histogram Equalization: https://simpleitk.readthedocs.io/en/master/link_SliceBySliceDecorator_docs.html
# 2) Bias Field Correction: https://medium.com/@alexandro.ramr777/how-to-do-bias-field-correction-with-python-156b9d51dd79
# 3) Denoising Filter: https://simpleitk.readthedocs.io/en/master/filters.html
# 4) Intensity normalization: https://medium.com/@sumitgulati59/brain-tumor-segmentation-b97de6619e04 (Nice Link)