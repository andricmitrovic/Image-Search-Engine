import torch
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader

from model import CNN
from utils import Utils
from tqdm import tqdm
import numpy as np
import heapq as hq
from torchvision.transforms import transforms as T

class Node(object):
    def __init__(self, norm, image):
        self.norm = norm
        self.image = image

    def __repr__(self):
        return self.image

    def __lt__(self, other):
        return self.norm < other.norm

    def __eq__(self, other):
        return self


def testLoop(cnn : CNN, ut : Utils):

    with torch.no_grad():

        criterion = ut.tripletLoss

        i = 0
        counter = []
        loss_history = []

        for data in tqdm(ut.testloader):

            i += 1

            img1 = data[0]
            img2 = data[1]
            img3 = data[2]

            img1, img2, img3 = img1.to(ut.device), img2.to(ut.device), img3.to(ut.device)

            output1 = torch.reshape(cnn(img1), (len(data[0]), 128))
            output2 = torch.reshape(cnn(img2), (len(data[0]), 128))
            output3 = torch.reshape(cnn(img3), (len(data[0]), 128))

            loss = criterion(output1, output2, output3)

            if i % 10 == 0:
                counter.append(i)
                loss_history.append(loss.item())

        print(np.mean(np.array(loss_history)))
        return counter, loss_history


if __name__ == '__main__':

    ut = Utils()

    path = "Models/30_12_2020_03_22"
    cnn = torch.load(path)
    cnn = cnn.to(ut.device)
    cnn.eval()

    #torch.manual_seed(9)

    transform_test = T.Compose([T.ToTensor(),
                                     T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                transform=transform_test)
    testloader = DataLoader(dataset,
                                 shuffle=True,
                                 num_workers=0,
                                 batch_size=1,
                                 pin_memory=True)

    anchor = None
    anchorEmbedding = None

    heap_list = []
    hq.heapify(heap_list)

    with torch.no_grad():
        for data in tqdm(testloader):

            image, _ = data

            input = image.to(ut.device)
            output = torch.reshape(cnn(input), (len(data[0]), 128))

            image = torch.squeeze(image, 0)

            if anchor is None:
                anchor = image
                anchorEmbedding = output
            else:
                norm = torch.dist(anchorEmbedding, output)
                hq.heappush(heap_list, Node(norm.item(), image))

        images = [anchor]
        for i in range(5):
            node = hq.heappop(heap_list)
            images += [node.image]

        ut.imshow(torchvision.utils.make_grid(images, nrow=3))

