import torch
import torch.utils.data
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt
from random import randrange




class Utils:

    def __init__(self, batchSize = 4, EPOCHS = 30):

        self.tranforms = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.tranforms)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=4,
                                                  shuffle=True, num_workers=0)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=self.tranforms)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=4,
                                                 shuffle=False, num_workers=0)

        self.datasetSize = len(self.trainset)
        self.EPOCHS = EPOCHS

        self.batchSize = batchSize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def getTripleImages(self):

        index = randrange(self.datasetSize)
        image1, label1 = self.trainset[index]

        index = randrange(self.datasetSize)
        image2, label2 = self.trainset[index]

        while True:
            index = randrange(self.datasetSize)
            image3, label3 = self.trainset[index]

            if label1 == label2:
                if label3 == label2:
                    continue
                else:
                    return [(image1, label1), (image2, label2), (image3, label3)]

            if label3 == label2:
                return [(image3, label3), (image2, label2), (image1, label1)]

            if label3 == label1:
                return [(image3, label3), (image1, label1), (image2, label2)]

    def getBatch(self):
        batch = []

        for _ in range(self.batchSize):
            batch += [self.getTripleImages()]

        return batch

    def displayBatch(self):
        outputs = self.getBatch()

        displayList = []

        for output in outputs:
            displayList += [output[0][0], output[1][0], output[2][0]]

        self.imshow(torchvision.utils.make_grid(displayList, nrow=3))

    def displayImgs(self, imgs):
        #output = self.getTripleImages()
        self.imshow(torchvision.utils.make_grid(imgs))

    def imshow(self, img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def tripletLoss(self):
        # L2 norm img1*img2
        # L2 norm img1*img3