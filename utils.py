import torch
import torch.utils.data
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from triplet_dataset import TripletDataset
from datetime import datetime



class Utils:

    def __init__(self, batchSize = 4, EPOCHS = 30):

        """
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=self.tranforms)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=4,
                                                 shuffle=False, num_workers=0)
        """

        self.EPOCHS = EPOCHS
        self.batchSize = batchSize

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

        self.trainset = TripletDataset("train")
        self.trainloader = DataLoader(self.trainset,
                                shuffle=True,
                                num_workers=0,
                                batch_size=self.batchSize)

        self.testset = TripletDataset("test")
        self.testloader = DataLoader(self.testset,
                                shuffle=True,
                                num_workers=0,
                                batch_size=self.batchSize)

    def imshow(self, img):
        #img = img / 2 + 0.5  # TODO denormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def displayData(self, num=5):

        dataiter = iter(self.trainloader)
        batch = next(dataiter)

        displayList = []

        for i in range(num):
            displayList += [batch[0][i], batch[1][i], batch[2][i]]

        self.imshow(torchvision.utils.make_grid(displayList, nrow=3))

    def tripletLoss(self, output1, output2, output3):
        #print(output1.shape)

        norm12 = [torch.dist(out1, out2) for out1, out2 in zip(output1, output2)]
        norm13 = [torch.dist(out1, out3) for out1, out3 in zip(output1, output3)]

        #print("Norm12", norm12)
        #print("Norm13", norm13)

        stacked = [torch.stack([norm12[i], norm13[i]], dim=0) for i in range(len(norm12))]
        #print("Norms stacked", stacked)

        f = nn.Softmax(dim=0)
        softmax = [f(x) for x in stacked]
        #print("Softmax", softmax)

        mse = nn.MSELoss()
        mse_loss = [mse(softmax[i][0], softmax[i][1] - 1) for i in range(len(softmax))]
        #print("Mse", mse_loss)

        mse_stacked = torch.stack(mse_loss)
        #print("Mse stacked", mse_stacked)

        total_loss = torch.mean(mse_stacked)
        #print("Total loss", total_loss)

        return total_loss

    def plotLoss(self, counter_train, loss_history_train, counter_test, loss_history_test, shouldSave=False):

        fig, ax = plt.subplots(2)
        fig.suptitle('Train vs Test')

        ax[0].plot(counter_train, loss_history_train, label="Train")
        ax[0].grid(True)

        ax[1].plot(counter_test, loss_history_test, label="Test")
        ax[1].grid(True)

        if shouldSave:
            today = datetime.now()
            path = "./Figures/" + today.strftime('%H_%M_%S_%d_%m_%Y')
            plt.savefig(path)

        plt.show()

    """
    dogs : 4 8 6 9 2
    cats:  7 9 5 2 4
    dsad:  1 6 8 9 4

    1 1 ?           6000 * 6000 * 44000 * 10
    1 2 ?
    1 3 ?
    1 4 ?
    ...
    1 n ?
    
    """
