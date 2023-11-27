import torch
from network import Unet3D, softmax_dice, Dice, ProposedYnet
import os; os.system('')

class Solver(object):
    def __init__(self, config, train_loader, test_loader):
        self.model_type = config.model_type
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
        elif self.model_type == "ProposedYNet":
            self.net = ProposedYnet(image_size=64, slice_depth=64, image_patch_size=4, slice_depth_patch_size=4, dim=768, depth=1, heads=8, mlp_dim=768, channels=4, dim_head=64, num_classes=2)
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
                print(i)
            train_loss = train_loss / len(self.train_loader.sampler)
            train_loss_list.append(train_loss)
            # print(f"Train Epoch: {epoch}, Overall Loss: {train_loss} | L1 Dice : {dice_1_t / len(self.train_loader)} | L2 Dice : {dice_2_t / len(self.train_loader)} | L3 Dice : {dice_3_t / len(self.train_loader)}")
            print(f"Train Epoch: {epoch}, Overall Loss: {train_loss}")
            file_name = 'model_f' + str(epoch) + ".pth"
            print("Saving Model")
            torch.save(self.net.state_dict(), file_name)

    def test(self):
        self.net.load_state_dict(torch.load("model_f10.pth"))
        self.net.train(False)
        self.net.eval()
        with torch.no_grad():
            dice0 = dice1 = dice2 = dice3 = 0
            for i, (images, GT, __) in enumerate(self.test_loader):
                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = self.net(images)
                SR = torch.nn.functional.softmax(SR, dim=1)
                dice0 += Dice(SR[:, 0, :, :, :], (GT == 0).float())
                dice1 += Dice(SR[:, 1, :, :, :], (GT == 1).float())
                # dice2 += Dice(SR[:, 2, :, :, :], (GT == 2).float())
                # dice3 += Dice(SR[:, 3, :, :, :], (GT == 3).float())

        # print(f"Test: L0 Dice : {1 - (dice0 / len(self.test_loader))} | L1 Dice : {1 - (dice1 / len(self.test_loader))} | L2 Dice : {1 - (dice2 / len(self.test_loader))} | L3 Dice : {1 - (dice3 / len(self.test_loader))}")
        print(f"Test: L0 Dice : {1 - (dice0 / len(self.test_loader))} | L1 Dice : {1 - (dice1 / len(self.test_loader))}")