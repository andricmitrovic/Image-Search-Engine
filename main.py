import torch
import torch.utils.data
from model import CNN
from utils import Utils
from torch import optim
from tqdm import tqdm
from datetime import datetime
import numpy as np


def trainLoop(cnn: CNN, ut: Utils, shouldSave=False):
    criterion = ut.tripletLoss
    if ut.optim == "SGD":
        optimizer = optim.SGD(cnn.parameters(), lr=ut.lr)
    elif ut.optim == "Adam":
        optimizer = optim.Adam(cnn.parameters(), lr=ut.lr)
    else:
        print("Wrong optim")
        return

    i = 0
    counter = []
    loss_history = []
    epochsSinceLastImprovement = 0
    bestLoss = None
    bestModel = cnn.state_dict()        # nothing smart at start

    #print(f"Epoch: {0}/{ut.EPOCHS}")
    for epoch in range(0, ut.EPOCHS):

        epochLoss = 0

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

            epochLoss += loss

            if i % 100 == 0:
                counter.append(i)
                loss_history.append(loss.item())

        epochLoss = epochLoss/len(ut.trainloader)

        if bestLoss is None:
            bestLoss = epochLoss

        if epochLoss >= bestLoss:
            epochsSinceLastImprovement += 1
            if epochsSinceLastImprovement >= 10:
                print("Early stopped!")
                ut.earlyStoppedAtEpoch = epoch
                cnn.load_state_dict(bestModel)
                break
        else:
            epochsSinceLastImprovement = 0
            bestLoss = epochLoss
            bestModel = cnn.state_dict()

    if shouldSave:
        today = datetime.now()
        path = "./Models/" + today.strftime('%d_%m_%Y_%H_%M')
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



        print(np.mean(np.array(loss_history)))
        return counter, loss_history


if __name__ == '__main__':

    ut = Utils(EPOCHS=1, batchSize=64, learning_rate=0.005, optim="SGD", lastLayerActivation="Sigmoid")
    #ut = Utils(EPOCHS=10, batchSize=64, learning_rate=0.005, weight_decay=1)
    cnn = CNN()
    cnn = cnn.to(ut.device)

    #ut.displayData()

    counter_train, loss_history_train = trainLoop(cnn, ut, shouldSave=True)
    counter_test, loss_history_test = testLoop(cnn, ut)

    ut.plotLoss(counter_train, loss_history_train, counter_test, loss_history_test, shouldSave=True)

