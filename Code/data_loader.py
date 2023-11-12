from torch.utils import data
from preprocess import readNpreprocessimage
import numpy as np

class BrainTumorSegDataset(data.Dataset):
    def __init__(self, imgs_pth, lbls_pth, config, mode):
        self.config = config
        self.mode = mode
        self.imgs_pth = imgs_pth
        self.lbls_pth = lbls_pth


    def __getitem__(self, index):
        img_pth = self.imgs_pth[index]
        lbl_pth = self.lbls_pth[index]
        images = readNpreprocessimage(img_pth, mask=False)
        mask_lbl = readNpreprocessimage(lbl_pth, mask=True)
        if lbl_pth['survivaldays'] is not None:
            survival_lbl = int(lbl_pth['survivaldays'])
        else:
            survival_lbl = np.NaN
        return images, mask_lbl, survival_lbl

    def __len__(self):
        return len(self.imgs_pth)

def get_loader(config, imgs_pth, lbls_pth, mode):
    dataset = BrainTumorSegDataset(imgs_pth, lbls_pth, config, mode)
    #img, mask, lbl = dataset[0]
    data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True)
    return data_loader