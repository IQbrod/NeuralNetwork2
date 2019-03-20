import torch.nn as nn
import torch.nn.functional as F

#Creation du réseau
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 30, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(30, 300, 3)
        self.fc1 = nn.Linear(300 * 13 * 13, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 300 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def op_counter(self,data,display):
        return 0