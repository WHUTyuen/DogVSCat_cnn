import torch.nn as nn
import torch

class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(100*100*3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )



    def forward(self, input):

        input = torch.reshape(input, (-1, 100*100*3))
        return self.layer1(input)
