import SimpleITK as sitk
import numpy as np
import torchvision.transforms as transforms
import torch


def readNpreprocessimage(imgs_pth, mask=False):
    finimage = {}
    if not mask:
        for keys in imgs_pth:
            temp = sitk.Cast(sitk.RescaleIntensity(sitk.ReadImage(imgs_pth[keys]), 0, 255), sitk.sitkUInt8)
            # Can fit in different preprocessing if required
            temp = np.moveaxis(sitk.GetArrayFromImage(temp), 0, -1)
            temp = transforms.ToTensor()(temp)
            finimage[keys] = temp
    else:
        maskimage = np.moveaxis(sitk.GetArrayFromImage(sitk.ReadImage(imgs_pth['mask'])), 0, -1)
        maskimage[maskimage == 4] = 3
        maskimage = maskimage.astype(np.uint8)
        maskimage = torch.from_numpy(maskimage).long()
        finimage['mask'] = maskimage
    return finimage


#Preprocessing Steps
# 1) Histogram Equalization: https://simpleitk.readthedocs.io/en/master/link_SliceBySliceDecorator_docs.html
# 2) Bias Field Correction: https://medium.com/@alexandro.ramr777/how-to-do-bias-field-correction-with-python-156b9d51dd79
# 3) Denoising Filter: https://simpleitk.readthedocs.io/en/master/filters.html
# 4) Intensity normalization: https://medium.com/@sumitgulati59/brain-tumor-segmentation-b97de6619e04 (Nice Link)