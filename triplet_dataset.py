from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision
from random import randrange


class TripletDataset(Dataset):

    def __init__(self):

        self.tranforms = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                     download=True, transform=self.tranforms)

        self.datasetSize = len(self.trainset)

    def __getitem__(self, idx):

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

    def __len__(self):
        return self.datasetSize
