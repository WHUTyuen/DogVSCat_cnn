import torch.nn as nn
import torch

class CNN_Net(nn.Module):

    def __init__(self):
        super().__init__()
        # 100
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(3,16,3,1),
            nn.ReLU(),
            nn.Conv2d(16,16,3,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(64, 80, 3, 1),
            nn.ReLU()
        )
        #  MAX MAX MAX
        self.mlp_layer = nn.Sequential(
            nn.Linear(8*8*80,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # N C H W
        out = self.cnn_layer(x)
        out = torch.reshape(out,(-1,8*8*80))
        return self.mlp_layer(out)
