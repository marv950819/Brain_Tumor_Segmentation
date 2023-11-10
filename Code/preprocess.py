import SimpleITK as sitk



def readNpreprocessimage(imgs_pth, mask=False):
    pass


#Preprocessing Steps
# 1) Histogram Equalization: https://simpleitk.readthedocs.io/en/master/link_SliceBySliceDecorator_docs.html
# 2) Bias Field Correction: https://medium.com/@alexandro.ramr777/how-to-do-bias-field-correction-with-python-156b9d51dd79
# 3) Denoising Filter: https://simpleitk.readthedocs.io/en/master/filters.html
# 4) Intensity normalization: https://medium.com/@sumitgulati59/brain-tumor-segmentation-b97de6619e04 (Nice Link)