import torch
from network import Unet3D, softmax_dice, Dice, ProposedVnet
import os; os.system('')
import imageio
import numpy as np

class Solver(object):
    def __init__(self, config, train_loader, test_loader):
        self.model_type = config.model_type
        self.image_size = config.image_size
        self.image_patch_size = config.image_patch_size
        self.slice_depth_size = config.slice_depth_size
        self.slice_depth_patch_size = config.slice_depth_patch_size
        self.channels = config.channels
        self.classes = config.classes
        self.survial_classes = config.survival_classes
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = None
        self.optimizer = None
        self.criterion = softmax_dice().to(self.device)
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.num_epochs = config.num_epochs
        self.build_model()

    def build_model(self):
        if self.model_type == 'UNet3D':
            self.net = Unet3D(in_dim=4, out_dim=4, num_filters=64)
        elif self.model_type == "ProposedVNet":
            self.net = ProposedVnet(image_size=self.image_size, slice_depth_size=self.slice_depth_size, image_patch_size=self.image_patch_size, slice_depth_patch_size=self.slice_depth_patch_size, dim=768, depth=1, heads=8, mlp_dim=768, channels=self.channels, dim_head=64, num_classes=self.classes, survival_classes=self.survial_classes)
        else:
            print('Not in list')
            return -1

        self.optimizer = torch.optim.AdamW(list(self.net.parameters()), self.lr, betas=(self.beta1, self.beta2))
        self.net.to(self.device)

    # self.print_network(self.net, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def train(self):
        train_loss_list = []
        for epoch in range(self.num_epochs):
            self.net.train(True)
            dice_1_t = dice_2_t= dice_3_t = train_loss = 0
            for i, (images, GT, __) in enumerate(self.train_loader):
                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = self.net(images)
                # loss, dice1, dice2, dice3 = self.criterion(SR, GT)
                loss = self.criterion(SR, GT)
                train_loss += loss
                # dice_1_t += dice1
                # dice_2_t += dice2
                # dice_3_t += dice3
                # Backprop + optimize
                self.net.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print(i)
            train_loss = train_loss / len(self.train_loader.sampler)
            train_loss_list.append(train_loss)
            # print(f"Train Epoch: {epoch}, Overall Loss: {train_loss} | L1 Dice : {dice_1_t / len(self.train_loader)} | L2 Dice : {dice_2_t / len(self.train_loader)} | L3 Dice : {dice_3_t / len(self.train_loader)}")
            print(f"Train Epoch: {epoch}, Overall Loss: {train_loss}")
            file_name = 'model_f' + str(epoch) + ".pth"
            print("Saving Model")
            torch.save(self.net.state_dict(), file_name)

    def test(self):
        self.net.load_state_dict(torch.load("model_f29_core.pth"))
        self.net.train(False)
        self.net.eval()
        with torch.no_grad():
            dice0 = dice1 = dice2 = dice3 = 0
            for i, (images, GT, __) in enumerate(self.test_loader):
                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = self.net(images)
                SR = torch.nn.functional.softmax(SR, dim=1)
                self.viz(images, GT, SR)
                dice0 += Dice(SR[:, 0, :, :, :], (GT == 0).float())
                dice1 += Dice(SR[:, 1, :, :, :], (GT == 1).float())
                # dice2 += Dice(SR[:, 2, :, :, :], (GT == 2).float())
                # dice3 += Dice(SR[:, 3, :, :, :], (GT == 3).float())

        # print(f"Test: L0 Dice : {1 - (dice0 / len(self.test_loader))} | L1 Dice : {1 - (dice1 / len(self.test_loader))} | L2 Dice : {1 - (dice2 / len(self.test_loader))} | L3 Dice : {1 - (dice3 / len(self.test_loader))}")
        print(f"Test: L0 Dice : {1 - (dice0 / len(self.test_loader))} | L1 Dice : {1 - (dice1 / len(self.test_loader))}")

    def viz(self, images, GT, SR):
        import matplotlib.pyplot as plt
        images = torch.permute(images, (0, 1, 3, 4, 2))
        GT = torch.permute(GT, (0, 2, 3, 1))
        SR = torch.permute(SR, (0, 1, 3, 4, 2))

        gif_images_GT = []
        gif_images_SR = []
        gif_combined = []
        for i in range(images.shape[4]):  
            slice_img = images[0, 0, :, :, i].cpu().numpy()
            slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())
            slice_img = (slice_img * 255).astype(np.uint8)
            slice_img_rgb = np.stack((slice_img,)*3, axis=-1)

            # overlay GT
            slice_gt = GT[0, :, :, i].cpu().numpy()
            slice_gt_colored = plt.get_cmap('jet')(slice_gt)[:, :, :3]
            slice_gt_colored = (slice_gt_colored * 255).astype(np.uint8)
            gt_mask = np.stack((slice_gt,)*3, axis=-1) > 0  # Create a 3D mask
            blended_img_GT = np.where(gt_mask, slice_gt_colored, slice_img_rgb.copy())
            blended_img_GT = blended_img_GT.astype(np.uint8)
            #gif_images_GT.append(blended_img_GT)

            #  overlay SR
            slice_sr = SR[0, 0, :, :, i].cpu().numpy()
            slice_sr_colored = plt.get_cmap('summer')(slice_sr)[:, :, :3]
            slice_sr_colored = (slice_sr_colored * 255).astype(np.uint8)
            sr_mask = np.stack((slice_sr,)*3, axis=-1) > 0.1

            blended_img_SR = np.where(sr_mask, slice_img_rgb, slice_sr_colored)
            blended_img_SR = blended_img_SR.astype(np.uint8)
            #gif_images_SR.append(blended_img_SR)

            gif_images_GT.append(blended_img_GT)
            gif_images_SR.append(blended_img_SR)

            # Concatenate images, GT, and SR side by side
            combined_img = np.concatenate((slice_img_rgb, blended_img_GT, blended_img_SR), axis=1)
            gif_combined.append(combined_img)

        # Save the combined GIF
        imageio.mimsave('results/output_combined_core.gif', gif_combined, duration=0.1)
           

    # def viz(self, images, GT, SR):
    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots(1, 6, figsize=(20, 5))
    #     slice = 64
    #     images = torch.permute(images, (0, 1, 3, 4, 2))
    #     GT = torch.permute(GT, (0, 2, 3, 1))
    #     SR = torch.permute(SR, (0, 1, 3, 4, 2))
    #     ax[0].imshow(images[0, 0, :, :, slice], cmap='gray')
    #     ax[0].set_title("T2-Flair")
    #     ax[1].imshow(images[0, 1, :, :, slice], cmap='gray')
    #     ax[1].set_title("T2")
    #     ax[2].imshow(images[0, 2, :, :, slice], cmap='gray')
    #     ax[2].set_title("T1ce")
    #     ax[3].imshow(images[0, 3, :, :, slice], cmap='gray')
    #     ax[3].set_title("T1")
    #     ax[4].imshow(GT[0, :, :, slice], cmap='gray')
    #     ax[4].set_title("Label Mask")
    #     ax[5].imshow(SR[0, 1, :, :, slice], cmap='gray')
    #     ax[5].set_title("Pred")
    #     plt.savefig(f"Result_whole_slice{str(slice)}.png", format='png', bbox_inches='tight')