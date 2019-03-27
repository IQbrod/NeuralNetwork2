import torch.nn as nn
import torch.nn.functional as F

#Creation du réseau
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 100, 3)
        self.conv4 = nn.Conv2d(100, 100, 3)
        self.fc1 = nn.Linear(100 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = x.view(-1, 100 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def op_counter(self,data,display):
        return 0