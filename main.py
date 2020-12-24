import torch
import torch.utils.data
import matplotlib.pyplot as plt
from model import CNN
from utils import Utils
from torch import optim
from tqdm import tqdm


"""
We used a momentum value of 0.9. We also used the dropout regularization technique with
p = 0.5 to avoid over-fitting
"""


def trainLoop(cnn : CNN, ut : Utils):
    criterion = ut.tripletLoss
    #optimizer = optim.SGD(cnn.parameters(), lr=0.5, weight_decay=0.9)
    optimizer = optim.SGD(cnn.parameters(), lr=0.005)
    #optimizer = optim.Adam(cnn.parameters(), lr=0.005)

    i = 0
    counter = []
    loss_history = []

    for epoch in range(0, 1):#ut.EPOCHS):
        for data in tqdm(ut.trainloader):
            #print(i, len(data), data[0].shape)

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
    return (counter, loss_history)


def testLoop(cnn : CNN, ut : Utils):
    with torch.no_grad():
        criterion = ut.tripletLoss

        i = 0
        counter = []
        loss_history = []

        for data in tqdm(ut.testloader):
            #print(i, len(data), data[0].shape)

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

        return (counter, loss_history)


if __name__ == '__main__':

    ut = Utils(batchSize=32)
    cnn = CNN()
    cnn = cnn.to(ut.device)

    counter_train, loss_history_train = trainLoop(cnn, ut)
    counter_test, loss_history_test = testLoop(cnn, ut)

    plt.plot(counter_train, loss_history_train, label="Train")
    plt.plot(counter_test, loss_history_test, label="Test")
    plt.grid(True)
    plt.title("Train vs Test")
    plt.show()

    """
    dataiter = iter(ut.dataloader)
    example_batch = next(dataiter)

    ut.displayBatch(example_batch)
    """


