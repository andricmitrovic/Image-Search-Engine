import torch
import torch.utils.data
import torchvision

from model import CNN
from utils import Utils
from torch import optim
from tqdm import tqdm
from datetime import datetime
import numpy as np
from random import randrange
import torch.nn as nn
import heapq as hq

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

    ut = Utils(EPOCHS=30, batchSize=64, learning_rate=0.005, optim="SGD", lastLayerActivation="Sigmoid")
    #ut = Utils(EPOCHS=10, batchSize=64, learning_rate=0.005, weight_decay=1)
    cnn = CNN()
    cnn = cnn.to(ut.device)

    path = "Models/28_12_2020_15_33"
    cnn = torch.load(path)
    cnn.eval()


    index = randrange(ut.testset.datasetSize)
    test_image = ut.testset[index][0]

    #print(test_image)

    test_image = test_image.to(ut.device)
    anchor = torch.reshape(cnn(torch.reshape(test_image, (1, 3, 32, 32))), (1, 128))

    heap_list = []
    hq.heapify(heap_list)

    for data in tqdm(ut.testloader):

        img1 = data[0]
        img2 = data[1]
        img3 = data[2]

        img1, img2, img3 = img1.to(ut.device), img2.to(ut.device), img3.to(ut.device)

        output1 = torch.reshape(cnn(img1), (len(data[0]), 128))
        output2 = torch.reshape(cnn(img2), (len(data[0]), 128))
        output3 = torch.reshape(cnn(img3), (len(data[0]), 128))

        norm1 = [torch.dist(anchor, out1) for out1 in output1]
        norm2 = [torch.dist(anchor, out2) for out2 in output2]
        norm3 = [torch.dist(anchor, out3) for out3 in output3]

        #print(img3[0])

        pairwise_distance = nn.PairwiseDistance(p=2)

        for i in range(len(norm1)):
            if int(torch.dist(test_image, img1[i]).item()) == 0:
                continue
            hq.heappush(heap_list, Node(norm1[i].item(), img1[i]))


        for i in range(len(norm2)):
            if int(torch.dist(test_image, img2[i]).item()) == 0:
                continue
            hq.heappush(heap_list, Node(norm2[i].item(), img2[i]))

        for i in range(len(norm3)):
            if int(torch.dist(test_image, img3[i]).item()) == 0:
                continue
            hq.heappush(heap_list, Node(norm3[i].item(), img3[i]))


    images =[test_image] + [node.image for node in hq.nlargest(5, heap_list)]

    for i in range(1, len(images)):
        print(torch.dist(images[0], images[i]).item())


    for i in range(len(images)):
        images[i] = images[i].to("cpu")

    ut.imshow(torchvision.utils.make_grid(images, nrow=3))


    #ut.accuracy(cnn)
