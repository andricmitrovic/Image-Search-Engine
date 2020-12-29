from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision
from random import randrange


class TripletDataset(Dataset):

    def __init__(self, dataset_type):

        self.dataset_type = dataset_type
        self.tranform_train = T.Compose([T.RandomCrop(32, padding=4),
                                    T.RandomHorizontalFlip(),
                                    T.ToTensor(),
                                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        self.transform_test = T.Compose([T.ToTensor(),
                                         T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        if self.dataset_type == "train":
            self.dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.tranform_train)
        else:
            self.dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform_test)

        self.datasetSize = len(self.dataset)

    def __getitem__(self, idx):

        index = randrange(self.datasetSize)
        image1, label1 = self.dataset[index]

        index = randrange(self.datasetSize)
        image2, label2 = self.dataset[index]

        while True:
            index = randrange(self.datasetSize)
            image3, label3 = self.dataset[index]

            if label1 == label2:
                if label3 == label2:
                    continue
                else:
                    return [image1, image2, image3]

            if label3 == label2:
                return [image3, image2, image1]

            if label3 == label1:
                return [image3, image1, image2]

    def __len__(self):
        return self.datasetSize
