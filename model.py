import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),                          # inplace=True ???
            nn.Dropout(p=0.5),                  # disable while testing

            nn.Conv2d(64, 128, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Conv2d(128, 256, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Conv2d(256, 128, 2),
            nn.Sigmoid(),                          # mozdad aktivaciona na [0,1] intervalu
            #nn.Dropout(p=0.5),     ???
        )

    def forward(self, input):
        output = self.cnn(input)
        return output