import torch
import torch.utils.data
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt
from random import randrange
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from triplet_dataset import TripletDataset



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

        self.dataset = TripletDataset()

        self.dataloader = DataLoader(self.dataset,
                                shuffle=True,
                                num_workers=0,
                                batch_size=self.batchSize)

    def imshow(self, img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def displayBatch(self, batch):

        displayList = []

        for i in range(self.batchSize):
            displayList += [batch[0][0][i], batch[1][0][i], batch[2][0][i]]

        self.imshow(torchvision.utils.make_grid(displayList, nrow=3))

    def tripletLoss(self, output1, output2, output3):

        f = nn.Softmax(dim=0)

        norm12 = (output1 - output2).norm()
        norm13 = (output1 - output3).norm()

        #print("Norm12", norm12, norm12.requires_grad)
        #print("Norm13", norm13, norm13.requires_grad)

        stacked = torch.stack([norm12, norm13], dim=0)
        #print("Norms stacked", stacked, stacked.requires_grad)

        softmax = f(stacked)
        #print("Softmax", softmax, softmax.requires_grad)

        mse = nn.MSELoss()
        output = mse(softmax[0], softmax[1] - 1)

        #print("Mse", output, output.requires_grad)

        return output




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
