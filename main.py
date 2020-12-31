import torch
import torch.utils.data
from model import CNN
from utils import Utils
from torch import optim
from tqdm import tqdm
from datetime import datetime


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
    counterTrain = []
    lossHistoryTrain = []

    j = 0
    counterVal = []
    lossHistoryVal = []
    epochsSinceLastImprovement = 0
    bestLoss = None
    bestModel = cnn.state_dict()

    for epoch in range(0, ut.EPOCHS):

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

            if i % 80 == 0:
                counterTrain.append(i)
                lossHistoryTrain.append(loss.item())

        with torch.no_grad():

            validationLoss = 0

            for data in tqdm(ut.valloader):

                j += 1

                img1 = data[0]
                img2 = data[1]
                img3 = data[2]

                img1, img2, img3 = img1.to(ut.device), img2.to(ut.device), img3.to(ut.device)

                output1 = torch.reshape(cnn(img1), (len(data[0]), 128))
                output2 = torch.reshape(cnn(img2), (len(data[0]), 128))
                output3 = torch.reshape(cnn(img3), (len(data[0]), 128))

                loss = criterion(output1, output2, output3)

                validationLoss += loss

                if j % 10 == 0:
                    counterVal.append(j)
                    lossHistoryVal.append(loss.item())

            validationLoss = validationLoss / len(ut.valloader)

        if bestLoss is None:
            bestLoss = validationLoss

        if validationLoss >= bestLoss:
            epochsSinceLastImprovement += 1
            if epochsSinceLastImprovement >= 50:
                print(f"Early stopped at epoch {epoch}!")
                ut.earlyStoppedAtEpoch = epoch - 49
                cnn.load_state_dict(bestModel)
                break
        else:
            epochsSinceLastImprovement = 0
            bestLoss = validationLoss
            bestModel = cnn.state_dict()

    if shouldSave:
        today = datetime.now()
        path = "./Models/" + today.strftime('%d_%m_%Y_%H_%M')
        torch.save(cnn, path)

    return counterTrain, lossHistoryTrain, counterVal, lossHistoryVal


if __name__ == '__main__':

    ut = Utils(EPOCHS=500, batchSize=256, learning_rate=0.0005, optim="Adam", lastLayerActivation="Sigmoid")

    cnn = CNN()
    cnn = cnn.to(ut.device)

    #ut.displayData()

    counterTrain, lossHistoryTrain, counterVal, lossHistoryVal = trainLoop(cnn, ut, shouldSave=True)

    ut.plotLoss(counterTrain, lossHistoryTrain, counterVal, lossHistoryVal, shouldSave=True, shouldDisplay=False)

    ut.accuracy(cnn)
