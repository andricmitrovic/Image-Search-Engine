import torch
import torch.utils.data
from model import CNN
from utils import Utils
from torch import optim
from tqdm import tqdm
from datetime import datetime


def trainLoop(cnn : CNN, ut : Utils, shouldSave = False):
    criterion = ut.tripletLoss
    #optimizer = optim.SGD(cnn.parameters(), lr=0.5, weight_decay=0.9)
    optimizer = optim.SGD(cnn.parameters(), lr=0.005)

    i = 0
    counter = []
    loss_history = []

    for epoch in range(0, 1):#ut.EPOCHS):
        for data in tqdm(ut.trainloader):

            i += 1

            img1 = data[0]
            img2 = data[1]
            img3 = data[2]

            img1, img2, img3 = img1.to(ut.device), img2.to(ut.device), img3.to(ut.device)

            optimizer.zero_grad()

            output1 = torch.reshape(cnn(img1), (len(data[0]), 128))
            output2 = torch.reshape(cnn(img2), (len(data[0]), 128))
            output3 = torch.reshape(cnn(img3), (len(data[0]), 128))

            loss = criterion(output1, output2, output3)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                #print("Epoch number {}\n Current loss {}\n".format(epoch, loss.item()))
                counter.append(i)
                loss_history.append(loss.item())

    #plt.plot(counter, loss_history)
    #plt.show()

    if shouldSave:
        today = datetime.now()
        path = "./Models/" + today.strftime('%H_%M_%S_%d_%m_%Y')
        torch.save(cnn, path)

    return counter, loss_history


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

        return counter, loss_history


if __name__ == '__main__':

    ut = Utils(batchSize=32)
    cnn = CNN()
    cnn = cnn.to(ut.device)

    # ut.displayData()

    counter_train, loss_history_train = trainLoop(cnn, ut, shouldSave=True)
    counter_test, loss_history_test = testLoop(cnn, ut)

    ut.plotLoss(counter_train, loss_history_train, counter_test, loss_history_test, shouldSave=True)

