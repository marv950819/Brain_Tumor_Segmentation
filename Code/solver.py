import torch
from network import Unet3D, softmax_dice
import os; os.system('')

class Solver(object):
    def __init__(self, config, train_loader):
        self.model_type = config.model_type
        self.train_loader = train_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = None
        self.optimizer = None
        self.criterion = softmax_dice().to(self.device)
        self.lr = config.lr
        self.num_epochs = config.num_epochs
        self.build_model()

    def build_model(self):
        if self.model_type == 'UNet3D':
            self.net = Unet3D(in_dim=4, out_dim=4, num_filters=64)
        else:
            print('Not in list')
            return -1

        self.optimizer = torch.optim.Adam(list(self.net.parameters()), self.lr)
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
                loss, dice1, dice2, dice3 = self.criterion(SR, GT)
                train_loss += loss
                dice_1_t += dice1
                dice_2_t += dice2
                dice_3_t += dice3
                # Backprop + optimize
                self.net.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_loss = train_loss / len(self.train_loader.sampler)
            train_loss_list.append(train_loss)
            print(f"Train Epoch: {epoch}, Overall Loss: {train_loss / len(self.train_loader)} | L1 Dice : {dice_1_t / len(self.train_loader)} | L2 Dice : {dice_2_t / len(self.train_loader)} | L3 Dice : {dice_3_t / len(self.train_loader)}")
