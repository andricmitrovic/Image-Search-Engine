import torch
import torch.utils.data
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt
from random import randrange
from model import CNN
from utils import Utils
from torch import optim


"""
We used a momentum value of 0.9. We also used the dropout regularization technique with
p = 0.5 to avoid over-fitting
"""

def trainLoop(cnn : CNN, ut : Utils):
    criterion = ut.tripletLoss
    optimizer = optim.SGD(cnn.parameters(), lr=0.05)

    i = 0
    counter = []
    loss_history = []

    for epoch in range(0, ut.EPOCHS):
        for batchRuns in range(0, 100):
            batch = ut.getBatch()
            for data in batch:

                i += 1

                img1, label1 = data[0]
                img2, label2 = data[1]
                img3, label3 = data[2]

                img1 = torch.reshape(img1, shape=(1, 3, 32, 32))  # (128)
                img2 = torch.reshape(img2, shape=(1, 3, 32, 32))
                img3 = torch.reshape(img3, shape=(1, 3, 32, 32))

                img1, img2, img3 = img1.to(ut.device), img2.to(ut.device), img3.to(ut.device)

                optimizer.zero_grad()
                output1 = torch.reshape(cnn(img1), (128,))
                output2 = torch.reshape(cnn(img2), (128,))
                output3 = torch.reshape(cnn(img3), (128,))

                loss = criterion(output1, output2, output3)
                loss.backward()
                optimizer.step()

                if i % 10 == 0:
                    print("Epoch number {}\n Current loss {}\n".format(epoch, loss.item()))
                    counter.append(i)
                    loss_history.append(loss.item())

    plt.plot(counter, loss_history)
    plt.show()

# for epoch in range(0,Config.train_number_epochs):
#     for i, data in enumerate(train_dataloader,0):
#         img0, img1 , label = data
#         img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
#         optimizer.zero_grad()
#         output1,output2 = net(img0,img1)
#         loss_contrastive = criterion(output1,output2,label)
#         loss_contrastive.backward()
#         optimizer.step()
#         if i %10 == 0 :
#             print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
#             iteration_number +=10
#             counter.append(iteration_number)
#             loss_history.append(loss_contrastive.item())
# show_plot(counter,loss_history)


if __name__ == '__main__':

    ut = Utils(batchSize=4)
    cnn = CNN()
    cnn = cnn.to(ut.device)
    #ut.displayBatch()
    #
    # inputBatch = ut.getBatch()
    #
    # img = inputBatch[0][0][0]
    #
    # img = torch.reshape(img, shape=(1, 3, 32, 32))
    #
    # cnn = CNN()
    # output = cnn(img)
    #
    # print(output.shape)

    trainLoop(cnn, ut)


